"""Parse Harbor CyberGym verifier output into exit codes and submissions."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rllm.integrations.cybergym.constants import DEFAULT_SCORING_MODE, SCORING_ANY_OF, SCORING_FINAL, ScoringMode
from rllm.integrations.cybergym.grade import DualBinaryGrade, SubmissionExits, dual_binary_reward, score_submissions

logger = logging.getLogger(__name__)

_POC_HEADER = re.compile(r"^---\s+(?P<name>poc(?:_\d+)?)\s+---\s*$", re.MULTILINE)
_VUL_LINE = re.compile(r"vul_exit\s*=\s*(?P<code>-?\d+)")
_FIX_LINE = re.compile(r"fix_exit\s*=\s*(?P<code>-?\d+)")
_VERIFY_COUNT = re.compile(r"Verifying\s+(?P<n>\d+)\s+PoC")
_NO_POC = re.compile(r"No PoC files found")
_JSON_OBJECT = re.compile(r"\{[^{}]*\"(?:vul_exit_code|reward)\"[^{}]*\}", re.DOTALL)


@dataclass
class CyberGymScore:
    reward: float
    vul_exit_code: int | None
    fix_exit_code: int | None
    n_submissions: int
    scoring_mode: ScoringMode
    scored_poc: str | None = None
    vul_crashed: bool = False
    fix_safe: bool = False
    scoring_incomplete: bool = False
    harbor_reward: float | None = None
    submissions: list[SubmissionExits] = field(default_factory=list)

    def as_artifacts(self) -> dict[str, Any]:
        return {
            "cybergym_trial_ran": True,
            "cybergym_reward": self.reward,
            "cybergym_scoring": self.scoring_mode,
            "cybergym_scored_poc": self.scored_poc,
            "cybergym_scoring_incomplete": self.scoring_incomplete,
            "vul_exit_code": self.vul_exit_code,
            "fix_exit_code": self.fix_exit_code,
            "n_submissions": self.n_submissions,
            "vul_crashed": self.vul_crashed,
            "fix_safe": self.fix_safe,
            "harbor_reward": self.harbor_reward if self.harbor_reward is not None else self.reward,
        }


def parse_reward_json(text: str) -> dict[str, Any] | None:
    """Extract the verify.py JSON object from verifier stdout or a log dump."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and ("vul_exit_code" in data or "reward" in data):
                return data
        except json.JSONDecodeError:
            pass
    for match in _JSON_OBJECT.finditer(text):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and ("vul_exit_code" in data or "reward" in data):
            return data
    return None


def parse_test_output(text: str) -> list[SubmissionExits]:
    """Parse Harbor ``tests/test.sh`` ``--- poc_NNN ---`` blocks."""
    if not text:
        return []
    headers = list(_POC_HEADER.finditer(text))
    if not headers:
        return []
    submissions: list[SubmissionExits] = []
    for idx, match in enumerate(headers):
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[match.end() : end]
        vul = _VUL_LINE.search(block)
        fix = _FIX_LINE.search(block)
        submissions.append(
            SubmissionExits(
                name=match.group("name"),
                vul_exit_code=int(vul.group("code")) if vul else None,
                fix_exit_code=int(fix.group("code")) if fix else None,
            )
        )
    return submissions


def infer_n_submissions(text: str, submissions: list[SubmissionExits]) -> int:
    if _NO_POC.search(text or ""):
        return 0
    count = _VERIFY_COUNT.search(text or "")
    if count:
        return int(count.group("n"))
    return len(submissions)


def _read_trial_logs(trial_uri: str) -> str:
    """Best-effort read of Harbor verifier logs next to the trial URI."""
    chunks: list[str] = []
    root = Path(trial_uri)
    candidates = [
        root / "logs" / "verifier" / "reward.json",
        root / "logs" / "verifier" / "test-output.txt",
        root / "verifier" / "reward.json",
        root / "verifier" / "test-output.txt",
    ]
    for path in candidates:
        try:
            if path.is_file():
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _verifier_blob(raw_result: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if not raw_result:
        return {}, ""
    vr = raw_result.get("verifier_result") or {}
    if not isinstance(vr, dict):
        return {}, str(vr)
    rewards = vr.get("rewards") if isinstance(vr.get("rewards"), dict) else {}
    stdout = vr.get("stdout") or vr.get("output") or vr.get("stderr") or ""
    return rewards, str(stdout)


def score_from_harbor_result(
    *,
    raw_result: dict[str, Any] | None,
    harbor_reward: float | None,
    trial_uri: str | None = None,
    scoring_mode: ScoringMode = DEFAULT_SCORING_MODE,
    extra_log: str = "",
) -> CyberGymScore:
    """Build a CyberGym score from a Harbor trial result.

    Default ``any_of`` uses Harbor ``test.sh`` / ``reward.json`` (adapter parity).
    ``final_submission`` recomputes from the last evaluated PoC and raises if
    Harbor short-circuited before verifying that file.
    """
    rewards, stdout = _verifier_blob(raw_result)
    disk_log = _read_trial_logs(trial_uri) if trial_uri else ""
    combined = "\n".join(part for part in (stdout, extra_log, disk_log) if part)

    submissions = parse_test_output(combined)
    reward_json = parse_reward_json(combined)
    if reward_json is None and rewards:
        if "vul_exit_code" in rewards or "reward" in rewards:
            reward_json = dict(rewards)

    n_submissions = infer_n_submissions(combined, submissions)
    if _NO_POC.search(combined) and not submissions:
        return CyberGymScore(
            reward=0.0,
            vul_exit_code=None,
            fix_exit_code=None,
            n_submissions=0,
            scoring_mode=scoring_mode,
            harbor_reward=0.0 if harbor_reward is None else harbor_reward,
        )

    if scoring_mode == SCORING_FINAL:
        grade = score_submissions(submissions, scoring_mode=SCORING_FINAL)
        return _from_grade(grade, n_submissions or len(submissions), scoring_mode, harbor_reward)

    # any_of: prefer the Harbor verifier's own reward + exit codes.
    if reward_json is not None and "vul_exit_code" in reward_json and "fix_exit_code" in reward_json:
        grade = dual_binary_reward(int(reward_json["vul_exit_code"]), int(reward_json["fix_exit_code"]))
        reward = float(reward_json["reward"]) if reward_json.get("reward") is not None else grade.reward
        if harbor_reward is not None:
            reward = float(harbor_reward)
        return CyberGymScore(
            reward=reward,
            vul_exit_code=grade.vul_exit_code,
            fix_exit_code=grade.fix_exit_code,
            n_submissions=n_submissions or max(len(submissions), 1),
            scoring_mode=SCORING_ANY_OF,
            scored_poc=submissions[0].name if submissions and grade.reward == 1.0 else (submissions[-1].name if submissions else None),
            vul_crashed=grade.vul_crashed,
            fix_safe=grade.fix_safe,
            harbor_reward=harbor_reward if harbor_reward is not None else reward,
            submissions=submissions,
        )

    if submissions:
        grade = score_submissions(submissions, scoring_mode=SCORING_ANY_OF)
        if harbor_reward is not None:
            grade = DualBinaryGrade(**{**grade.__dict__, "reward": float(harbor_reward)})
        return _from_grade(grade, n_submissions or len(submissions), SCORING_ANY_OF, harbor_reward)

    if harbor_reward is not None:
        return CyberGymScore(
            reward=float(harbor_reward),
            vul_exit_code=None,
            fix_exit_code=None,
            n_submissions=n_submissions,
            scoring_mode=SCORING_ANY_OF,
            harbor_reward=float(harbor_reward),
        )
    return CyberGymScore(
        reward=0.0,
        vul_exit_code=None,
        fix_exit_code=None,
        n_submissions=n_submissions,
        scoring_mode=scoring_mode,
        harbor_reward=0.0,
    )


def _from_grade(
    grade: DualBinaryGrade,
    n_submissions: int,
    scoring_mode: ScoringMode,
    harbor_reward: float | None,
) -> CyberGymScore:
    return CyberGymScore(
        reward=grade.reward,
        vul_exit_code=grade.vul_exit_code,
        fix_exit_code=grade.fix_exit_code,
        n_submissions=n_submissions,
        scoring_mode=scoring_mode,
        scored_poc=grade.scored_poc,
        vul_crashed=grade.vul_crashed,
        fix_safe=grade.fix_safe,
        harbor_reward=harbor_reward if harbor_reward is not None else grade.reward,
    )
