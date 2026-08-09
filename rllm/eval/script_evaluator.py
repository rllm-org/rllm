"""ShellScriptEvaluator: run a shell verifier inside a sandbox.

Used when ``dataset.toml`` / ``task.toml`` declares ``[verifier].script``
or when ``tests/test.sh`` is auto-detected. Implements the rLLM
:class:`~rllm.types.Evaluator` protocol.

Reward contract (Harbor-compatible): the script writes to one of
``/tmp/rllm/reward.json``, ``/logs/verifier/reward.json``, or
``/logs/verifier/reward.txt``. The first existing file wins.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
from pathlib import Path

from rllm.eval.types import EvalOutput, Signal
from rllm.sandbox.protocol import Sandbox
from rllm.types import Episode, Task

logger = logging.getLogger(__name__)


# Reward file search order (first existing file wins)
_REWARD_PATHS = [
    "/tmp/rllm/reward.json",
    "/logs/verifier/reward.json",
    "/logs/verifier/reward.txt",
]

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REMOTE_ENV_PATH = "/tmp/rllm/verifier.env"


class ShellScriptEvaluator:
    """Run a verifier script inside the sandbox, parse the reward file.

    Constructed by :func:`rllm.eval._resolution._resolve_evaluator` once
    the sandbox is alive — the evaluator carries its sandbox reference
    internally instead of fishing it out of episode artifacts.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        script_path: str = "tests/test.sh",
        verifier_user: str | None = None,
        verifier_timeout: float = 600.0,
        reward_file_override: str | None = None,
        verifier_env: dict[str, str] | None = None,
    ):
        self.sandbox = sandbox
        self.script_path = script_path  # relative to the task's directory
        self.verifier_user = verifier_user
        self.verifier_timeout = verifier_timeout
        self.reward_file_override = reward_file_override
        self.verifier_env = dict(verifier_env or {})

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        tests_dir = task.task_dir / Path(self.script_path).parent
        script_name = Path(self.script_path).name
        if not tests_dir.is_dir():
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                metadata={"error": f"no {tests_dir} directory"},
            )

        v_user = self.verifier_user

        # Prepare reward directories
        try:
            self.sandbox.exec("mkdir -p /tmp/rllm /logs/verifier", timeout=10, user=v_user)
        except Exception:
            pass

        # Upload to /tests/ (Harbor convention — scripts may reference /tests/*.py)
        self.sandbox.upload_dir(str(tests_dir), "/tests")

        if not (tests_dir / script_name).exists():
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                metadata={"error": f"verifier script {script_name} not found in {tests_dir}"},
            )

        # Only ``cd`` when the task explicitly declared a workdir.
        # Otherwise the Dockerfile's WORKDIR wins — required for swesmith
        # and similar harbor task families whose verifier scripts run
        # ``git`` and ``pytest`` against ``/testbed`` (the image WORKDIR)
        # and silently collect zero tests when forced into ``/workspace``.
        workdir = task.metadata.get("workdir")
        cd_prefix = f"cd {workdir} && " if workdir else ""
        try:
            resolved_env = _resolve_verifier_env(self.verifier_env)
        except ValueError as e:
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                metadata={"error": str(e), "ungraded": True},
            )

        env_prefix = ""
        try:
            if resolved_env:
                self._upload_verifier_env(resolved_env, v_user)
                env_prefix = f"set -a && . {_REMOTE_ENV_PATH} && set +a && "
            self.sandbox.exec(
                f"chmod +x /tests/{script_name} && {env_prefix}{cd_prefix}/tests/{script_name}",
                timeout=self.verifier_timeout,
                user=v_user,
            )
        except Exception as e:
            # Verifier exit != 0 is the *expected* outcome when an agent
            # didn't solve the task; the reward (read from reward.txt
            # below) carries the signal. Log at debug so a benchmark of
            # 100 unsolved tasks doesn't spam 100 multi-KB stack traces.
            logger.debug("Verifier exited non-zero for %s: %s", task.id, e)
        finally:
            if resolved_env:
                try:
                    self.sandbox.exec(f"rm -f {_REMOTE_ENV_PATH}", timeout=10, user=v_user)
                except Exception:
                    logger.debug("Could not remove verifier environment file for %s", task.id)

        # Read reward (as verifier — agent may not have read access)
        reward_paths = list(_REWARD_PATHS)
        if self.reward_file_override:
            reward_paths.insert(0, self.reward_file_override)
        return _read_reward_from_sandbox(self.sandbox, reward_paths, user=v_user)

    def _upload_verifier_env(self, env: dict[str, str], verifier_user: str | None) -> None:
        body = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items()) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="rllm-verifier-", suffix=".env") as handle:
            handle.write(body)
            handle.flush()
            self.sandbox.upload_file(handle.name, _REMOTE_ENV_PATH)

        owner = shlex.quote(verifier_user) if verifier_user else None
        ownership = f"chown {owner} {_REMOTE_ENV_PATH} && " if owner else ""
        self.sandbox.exec(f"{ownership}chmod 600 {_REMOTE_ENV_PATH}", timeout=10)


def _resolve_verifier_env(config: dict[str, str]) -> dict[str, str]:
    """Resolve ``${HOST_VAR}`` references without exposing values to the agent."""
    resolved: dict[str, str] = {}
    missing: set[str] = set()
    for key, raw_value in config.items():
        if not _ENV_NAME.fullmatch(key):
            raise ValueError(f"invalid verifier environment variable name: {key!r}")
        if not isinstance(raw_value, str):
            raise ValueError(f"verifier environment value for {key} must be a string")

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            value = os.environ.get(name)
            if value is None:
                missing.add(name)
                return ""
            return value

        resolved[key] = _ENV_REFERENCE.sub(replace, raw_value)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing required verifier environment variable(s): {names}")
    return resolved


# ---------------------------------------------------------------------------
# Reward parsing helpers (extracted from rllm/tasks/task.py)
# ---------------------------------------------------------------------------


def _read_reward_from_sandbox(sandbox: Sandbox, paths: list[str], user: str | None = None) -> EvalOutput:
    """Try reading reward from the sandbox at each path in order."""
    for path in paths:
        try:
            check = sandbox.exec(f"test -f {path} && echo yes || echo no", timeout=10, user=user).strip()
            if check != "yes":
                continue
            raw = sandbox.exec(f"cat {path}", timeout=10, user=user).strip()
            if not raw:
                continue
            if path.endswith(".txt"):
                reward = float(raw)
                return EvalOutput(reward=reward, is_correct=reward >= 1.0)
            return _parse_reward_json(raw)
        except Exception as e:
            logger.debug("Could not read reward from %s: %s", path, e)
            continue

    logger.warning("No reward file found at any of: %s", paths)
    return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no reward file found"})


def _parse_reward_json(raw: str) -> EvalOutput:
    """Parse a JSON reward file into an EvalOutput.

    Supports both ``{"reward": 0.5}`` and Harbor-style ``{"rewards": {...}}``.
    """
    data = json.loads(raw)

    if "reward" in data:
        reward = float(data["reward"])
    elif "rewards" in data and data["rewards"]:
        reward = sum(float(v) for v in data["rewards"].values()) / len(data["rewards"])
    else:
        reward = 0.0

    is_correct = data.get("is_correct", reward >= 1.0)

    signals: list[Signal] = []
    for key, val in data.get("signals", {}).items():
        signals.append(Signal(name=key, value=float(val)))
    for key, val in data.get("rewards", {}).items():
        if key != "reward":
            signals.append(Signal(name=key, value=float(val)))

    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=signals,
        metadata=data.get("metadata", {}),
    )
