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
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rllm.eval.types import EvalOutput, Signal
from rllm.sandbox.protocol import Sandbox, SandboxCommandTimeout
from rllm.types import Episode, Task

logger = logging.getLogger(__name__)


# Sandbox repos are root-owned but may be inspected as another user, which trips
# git's dubious-ownership guard; every git call rLLM makes opts out of it.
_GIT = "git -c safe.directory='*'"

# Reward file search order (first existing file wins)
_REWARD_PATHS = [
    "/tmp/rllm/reward.json",
    "/logs/verifier/reward.json",
    "/logs/verifier/reward.txt",
]

# Keys of a reward file that carry the verdict itself; everything else in it is
# grader detail lifted to signals (see _parse_reward_json).
_RESERVED_REWARD_KEYS = frozenset({"reward", "rewards", "is_correct", "signals", "metadata"})

# Enough for a suite's failure summary without bloating results.json.
_VERIFIER_STDOUT_TAIL_CHARS = 8000


@dataclass(frozen=True)
class ArtifactSpec:
    """The TB3-relevant subset of Harbor's artifact declaration."""

    source: str
    service: str = "main"
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class CollectedArtifact:
    spec: ArtifactSpec
    local_path: Path
    is_dir: bool


def normalize_artifacts(entries: list[str | dict] | None) -> list[ArtifactSpec]:
    """Normalize Harbor's string and object artifact forms."""
    artifacts: list[ArtifactSpec] = []
    for entry in entries or []:
        if isinstance(entry, str):
            artifacts.append(ArtifactSpec(source=entry))
            continue
        if not isinstance(entry, dict) or not entry.get("source"):
            raise ValueError(f"invalid artifact declaration: {entry!r}")
        artifacts.append(
            ArtifactSpec(
                source=str(entry["source"]),
                service=str(entry.get("service") or "main"),
                exclude=tuple(str(pattern) for pattern in (entry.get("exclude") or [])),
            )
        )
    return artifacts


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
        git_heads: dict[str, str] | None = None,
        verifier_sandbox_factory: Callable[[], Sandbox] | None = None,
        verifier_tests_baked: bool = False,
        collect_commands: list[dict] | None = None,
        artifacts: list[str | dict] | None = None,
    ):
        self.sandbox = sandbox
        self.script_path = script_path  # relative to the task's directory
        self.verifier_user = verifier_user
        self.verifier_timeout = verifier_timeout
        self.reward_file_override = reward_file_override
        # repo root -> HEAD sha as of before the agent ran (see _restore_git_heads).
        self.git_heads = git_heads or {}
        # Set only for a task declaring [verifier].environment_mode = "separate":
        # grading then runs in a container this builds, not the agent's. See
        # :meth:`_grading_sandbox`.
        self.verifier_sandbox_factory = verifier_sandbox_factory
        self.verifier_tests_baked = verifier_tests_baked
        self.collect_commands = collect_commands or []
        self.artifacts = normalize_artifacts(artifacts)

    @property
    def separate_env(self) -> bool:
        """Whether grading runs in its own container rather than the agent's."""
        return self.verifier_sandbox_factory is not None

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        tests_dir = task.task_dir / Path(self.script_path).parent
        script_name = Path(self.script_path).name
        if not tests_dir.is_dir():
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="AddTestsDirError",
                metadata={"error": f"no {tests_dir} directory"},
            )

        v_user = self.verifier_user

        if not (tests_dir / script_name).exists():
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="AddTestsDirError",
                metadata={"error": f"verifier script {script_name} not found in {tests_dir}"},
            )

        # A separate-mode task grades in a fresh container; everything the agent
        # produced reaches it as collected artifacts. Own it for the duration so
        # the box can't outlive the grade.
        grading_sandbox = self.sandbox
        with tempfile.TemporaryDirectory(prefix="rllm-artifacts-") as staging:
            collected: list[CollectedArtifact] = []
            if self.separate_env:
                collected = self._run_collect(task, v_user, Path(staging))
                try:
                    grading_sandbox = self.verifier_sandbox_factory()  # type: ignore[misc]
                except Exception as e:
                    logger.warning("Could not create verifier sandbox for %s: %s", task.id, e)
                    return EvalOutput(
                        reward=0.0,
                        is_correct=False,
                        error="VerifierEnvironmentError",
                        metadata={"error": f"failed to create separate verifier environment: {e}"},
                    )
            try:
                return self._grade_in(grading_sandbox, task, tests_dir, script_name, v_user, collected)
            finally:
                if grading_sandbox is not self.sandbox:
                    try:
                        grading_sandbox.close()
                    except Exception:
                        logger.debug("Verifier sandbox teardown failed for %s", task.id, exc_info=True)

    def _grade_in(
        self,
        sandbox: Sandbox,
        task: Task,
        tests_dir: Path,
        script_name: str,
        v_user: str | None,
        collected: list[CollectedArtifact],
    ) -> EvalOutput:
        """Upload the tests (and any collected artifacts), run the verifier, read the reward."""
        # Prepare reward directories. Harbor mounts /logs as a writable volume, so
        # task verifiers assume any user can write under it — e.g. a test.sh that
        # runs ``su postgres -c "pg_ctl -l /logs/verifier/pg.log"`` after creating
        # the dir as root. A bare mkdir leaves 755 root:root and that write fails
        # with EACCES; 1777 (like /tmp) restores harbor's contract. Grading always
        # runs after the agent has finished, so this widens nothing the agent sees.
        try:
            sandbox.exec("mkdir -p /tmp/rllm /logs/verifier && chmod 1777 /logs/verifier", timeout=10, user=v_user)
        except Exception:
            pass

        # Upload to /tests/ (Harbor convention — scripts may reference /tests/*.py).
        # A failed upload means the verifier can't run — a grading-infra failure,
        # not a task score; tag it so the engine doesn't read it as reward 0.
        if not self.verifier_tests_baked:
            try:
                sandbox.upload_dir(str(tests_dir), "/tests")
            except Exception as e:
                logger.warning("Failed to upload tests dir for %s: %s", task.id, e)
                return EvalOutput(
                    reward=0.0,
                    is_correct=False,
                    error="AddTestsDirError",
                    metadata={"error": f"failed to upload tests dir: {e}"},
                )

        # Re-materialise what the collect step snapshotted, at the same paths it
        # wrote them to — the verifier image doesn't pre-create those dirs.
        for artifact in collected:
            try:
                target_dir = artifact.spec.source if artifact.is_dir else str(PurePosixPath(artifact.spec.source).parent)
                sandbox.exec(
                    f"mkdir -p {shlex.quote(target_dir)} && chmod 777 {shlex.quote(target_dir)}",
                    timeout=30,
                    user=v_user,
                )
                if artifact.is_dir:
                    sandbox.upload_dir(str(artifact.local_path), artifact.spec.source)
                else:
                    sandbox.upload_file(str(artifact.local_path), artifact.spec.source)
            except Exception as e:
                logger.warning("Could not upload artifact %s for %s: %s", artifact.spec.source, task.id, e)

        # A task workdir belongs to the agent image. A separate verifier image
        # must retain its own Dockerfile WORKDIR; the agent path may not exist
        # there, which would prevent the verifier script from running at all.
        workdir = task.metadata.get("workdir") if sandbox is self.sandbox else None
        cd_prefix = f"cd {workdir} && " if workdir else ""

        # Only meaningful in the agent's own container; a fresh verifier box is
        # already at the image's commit.
        if sandbox is self.sandbox:
            self._restore_git_heads(v_user)

        verifier_stdout = ""
        try:
            verifier_stdout = sandbox.exec(
                f"chmod +x /tests/{script_name} && {cd_prefix}/tests/{script_name}",
                timeout=self.verifier_timeout,
                user=v_user,
            )
        except SandboxCommandTimeout as e:
            # The verifier itself blew its time budget — a grading-infra
            # failure, NOT "agent scored 0". Tag it (Harbor's VerifierTimeoutError)
            # so the engine routes it to VERIFIER_TIMEOUT and training drops the
            # untrustworthy reward instead of training on a spurious zero.
            logger.warning("Verifier timed out after %ss for %s: %s", self.verifier_timeout, task.id, e)
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="VerifierTimeoutError",
                metadata={"error": f"verifier timed out after {self.verifier_timeout}s"},
            )
        except Exception as e:
            # Verifier exit != 0 is the *expected* outcome when an agent
            # didn't solve the task; the reward (read from reward.txt
            # below) carries the signal. Log at debug so a benchmark of
            # 100 unsolved tasks doesn't spam 100 multi-KB stack traces.
            logger.debug("Verifier exited non-zero for %s: %s", task.id, e)
            # The backend raises with the output attached — the only copy we get
            # on the failure path, which is the one worth debugging.
            verifier_stdout = str(e)

        # Read reward (as verifier — agent may not have read access)
        reward_paths = list(_REWARD_PATHS)
        if self.reward_file_override:
            reward_paths.insert(0, self.reward_file_override)
        out = _read_reward_from_sandbox(sandbox, reward_paths, user=v_user)
        # Harbor verifiers cat every suite's raw log to stdout so "the reason a
        # test failed is never lost", and the sandbox dies at teardown — without
        # this a failed task records only a bare count (f2p 42/43) with no way to
        # tell which test failed. Tail, not head: the verdict is at the end.
        if verifier_stdout:
            out.metadata["verifier_stdout_tail"] = verifier_stdout[-_VERIFIER_STDOUT_TAIL_CHARS:]
        return out

    def _run_collect(self, task: Task, user: str | None, staging: Path) -> list[CollectedArtifact]:
        """Run ``[[verifier.collect]]`` in the agent's box and pull the artifacts out.

        Harbor's separate-mode contract: the agent's container is the only place
        the work exists, so it snapshots state into files (deepswe writes
        ``git diff --binary <base_commit> HEAD`` to ``model.patch``) which are then
        re-materialised in the verifier container. Best-effort per command — a
        collect step that fails leaves that artifact missing, which the verifier
        reports as an unsubmitted patch rather than a grading crash.
        """
        main_hooks = [entry for entry in self.collect_commands if _entry_service(entry) == "main"]
        sidecar_hooks = [entry for entry in self.collect_commands if _entry_service(entry) != "main"]
        main_artifacts = [artifact for artifact in self.artifacts if artifact.service == "main"]
        sidecar_artifacts = [artifact for artifact in self.artifacts if artifact.service != "main"]

        self._run_collect_hooks(task, main_hooks, user)
        collected = self._collect_artifacts(task, main_artifacts, staging, user)

        if sidecar_hooks or sidecar_artifacts:
            stop_service = getattr(self.sandbox, "stop_service", None)
            if not callable(stop_service):
                raise RuntimeError(f"task {task.id!r} declares sidecar artifacts but its sandbox is not service-capable")
            stop_service("main")
            self._run_collect_hooks(task, sidecar_hooks, user)
            collected.extend(self._collect_artifacts(task, sidecar_artifacts, staging, user, offset=len(collected)))
        return collected

    def _run_collect_hooks(self, task: Task, entries: list[dict], user: str | None) -> None:
        for entry in entries:
            command = entry.get("command") if isinstance(entry, dict) else str(entry)
            if not command:
                continue
            service = _entry_service(entry)
            try:
                _service_exec(self.sandbox, service, command, timeout=float((entry or {}).get("timeout_sec", 300.0)), user=user)
            except Exception as e:
                logger.warning("collect step failed for %s in %s (%s): %s", task.id, service, command[:80], e)

    def _collect_artifacts(
        self,
        task: Task,
        artifacts: list[ArtifactSpec],
        staging: Path,
        user: str | None,
        *,
        offset: int = 0,
    ) -> list[CollectedArtifact]:
        collected: list[CollectedArtifact] = []
        for index, artifact in enumerate(artifacts, start=offset):
            target = staging / str(index)
            try:
                kind = _service_exec(
                    self.sandbox,
                    artifact.service,
                    f"if [ -d {shlex.quote(artifact.source)} ]; then echo directory; elif [ -f {shlex.quote(artifact.source)} ]; then echo file; else echo missing; fi",
                    timeout=30,
                    user=user,
                ).strip()
                if kind.endswith("directory"):
                    local_path = _download_directory(self.sandbox, artifact, target, user=user)
                    collected.append(CollectedArtifact(artifact, local_path, True))
                elif kind.endswith("file"):
                    target.mkdir(parents=True, exist_ok=True)
                    local_path = target / (PurePosixPath(artifact.source.rstrip("/")).name or "artifact")
                    local_path.write_bytes(_service_download_file(self.sandbox, artifact.service, artifact.source))
                    mode = _remote_file_mode(self.sandbox, artifact, user=user)
                    if mode is not None:
                        os.chmod(local_path, mode)
                    collected.append(CollectedArtifact(artifact, local_path, False))
                else:
                    # Compatibility for simple Sandbox test doubles and older
                    # providers that cannot answer the shell type probe but can
                    # still download a regular file.
                    target.mkdir(parents=True, exist_ok=True)
                    local_path = target / (PurePosixPath(artifact.source.rstrip("/")).name or "artifact")
                    local_path.write_bytes(_service_download_file(self.sandbox, artifact.service, artifact.source))
                    collected.append(CollectedArtifact(artifact, local_path, False))
            except Exception as e:
                logger.warning("Could not collect artifact %s from %s for %s: %s", artifact.source, artifact.service, task.id, e)
        return collected

    def _restore_git_heads(self, user: str | None) -> None:
        """Move each repo's HEAD back to the commit it had before the agent ran.

        In-sandbox verifiers restore the task's official test files with
        ``git checkout HEAD -- <path>`` before applying their test patch, which
        assumes HEAD is still the *image's* commit. rLLM runs the verifier in the
        agent's own sandbox, so an agent that commits its work (mini-swe-agent
        and other git-oriented agents do) moves HEAD — and that "restore" then
        resurrects the agent's own edited/added test files, making the official
        test patch unappliable ("already exists in working directory" / "patch
        does not apply") and crashing the verifier before it grades anything.

        ``reset --soft`` moves the branch ref only: the working tree and index
        keep the agent's edits, which is exactly what the verifier grades. No-op
        when nothing was captured (``_capture_git_heads``) or HEAD never moved.
        """
        for root, sha in self.git_heads.items():
            cmd = f'cur=$({_GIT} -C {root} rev-parse HEAD 2>/dev/null); [ "$cur" = {sha} ] || {{ {_GIT} -C {root} reset --soft {sha} && echo moved; }}'
            try:
                if "moved" in self.sandbox.exec(cmd, timeout=60, user=user):
                    logger.info("Restored %s HEAD to pre-agent commit %s before grading", root, sha[:12])
            except Exception as e:
                logger.warning("Could not restore %s HEAD to %s: %s", root, sha[:12], e)


# ---------------------------------------------------------------------------
# Artifact transfer between the agent's box and a separate verifier box
# ---------------------------------------------------------------------------


def _entry_service(entry: dict | str) -> str:
    return str(entry.get("service") or "main") if isinstance(entry, dict) else "main"


def _service_exec(
    sandbox: Sandbox,
    service: str,
    command: str,
    *,
    timeout: float | None = None,
    user: str | None = None,
) -> str:
    if service == "main":
        return sandbox.exec(command, timeout=timeout, user=user)
    execute = getattr(sandbox, "service_exec", None)
    if not callable(execute):
        raise RuntimeError(f"sandbox cannot execute in sidecar service {service!r}")
    return execute(service, command, timeout=timeout, user=user)


def _service_download_file(sandbox: Sandbox, service: str, path: str) -> bytes:
    if service == "main":
        return sandbox.download_file(path)
    download = getattr(sandbox, "service_download_file", None)
    if not callable(download):
        raise RuntimeError(f"sandbox cannot download from sidecar service {service!r}")
    return download(service, path)


def _remote_file_mode(
    sandbox: Sandbox,
    artifact: ArtifactSpec,
    *,
    user: str | None = None,
) -> int | None:
    """Read a regular artifact's permission bits so executables stay executable."""
    path = shlex.quote(artifact.source)
    command = f"stat -c '%a' {path} 2>/dev/null || stat -f '%Lp' {path} 2>/dev/null"
    try:
        raw = _service_exec(sandbox, artifact.service, command, timeout=30, user=user).strip().splitlines()[-1]
        return int(raw, 8) & 0o7777
    except (IndexError, TypeError, ValueError, RuntimeError):
        return None


def _download_directory(
    sandbox: Sandbox,
    artifact: ArtifactSpec,
    target: Path,
    *,
    user: str | None = None,
) -> Path:
    """Download a remote directory as one filtered tar archive."""
    source = PurePosixPath(artifact.source.rstrip("/"))
    if not source.name:
        raise ValueError("artifact directory source must not be filesystem root")
    archive_path = f"/tmp/.rllm-artifact-{uuid.uuid4().hex}.tgz"
    excludes = " ".join(shlex.quote(f"--exclude={pattern}") for pattern in artifact.exclude)
    command = " ".join(
        part
        for part in (
            "tar czf",
            shlex.quote(archive_path),
            excludes,
            "-C",
            shlex.quote(str(source.parent)),
            shlex.quote(source.name),
        )
        if part
    )
    try:
        _service_exec(sandbox, artifact.service, command, timeout=600, user=user)
        archive = _service_download_file(sandbox, artifact.service, archive_path)
    finally:
        try:
            _service_exec(sandbox, artifact.service, f"rm -f {shlex.quote(archive_path)}", timeout=30, user=user)
        except Exception:
            pass

    target.mkdir(parents=True, exist_ok=True)
    local_archive = target / "artifact.tgz"
    local_archive.write_bytes(archive)
    with tarfile.open(local_archive, mode="r:gz") as tar:
        tar.extractall(target, filter="data")
    local_archive.unlink()
    local_path = target / source.name
    if not local_path.is_dir():
        raise FileNotFoundError(f"downloaded artifact archive did not contain {source.name!r}")
    return local_path


# ---------------------------------------------------------------------------
# Reward parsing helpers (extracted from rllm/tasks/task.py)
# ---------------------------------------------------------------------------


def _read_reward_from_sandbox(sandbox: Sandbox, paths: list[str], user: str | None = None) -> EvalOutput:
    """Try reading reward from the sandbox at each path in order.

    Distinguishes grading-infra failures from a legitimate ``reward=0`` by
    setting :attr:`EvalOutput.error` (Harbor-aligned names): an unparseable
    file → ``VerifierOutputParseError``, an empty file → ``RewardFileEmptyError``,
    no file at all → ``RewardFileNotFoundError``. The engine promotes these to
    ``GRADING_ERROR`` so the untrustworthy reward is filtered from training.
    """
    saw_empty = False
    for path in paths:
        try:
            check = sandbox.exec(f"test -f {path} && echo yes || echo no", timeout=10, user=user).strip()
            if check != "yes":
                continue
            raw = sandbox.exec(f"cat {path}", timeout=10, user=user).strip()
        except Exception as e:
            logger.debug("Could not read reward from %s: %s", path, e)
            continue
        if not raw:
            saw_empty = True
            continue
        try:
            if path.endswith(".txt"):
                reward = float(raw)
                out = EvalOutput(reward=reward, is_correct=reward >= 1.0)
            else:
                out = _parse_reward_json(raw)
        except Exception as e:
            logger.warning("Could not parse reward file %s: %s", path, e)
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="VerifierOutputParseError",
                metadata={"error": f"could not parse reward file {path}: {e}"},
            )
        # A negative reward is the harness's "no verdict" sentinel, not a score:
        # harbor-style test.sh wrappers trap a crashed grader and write -1 so the
        # failure stays visible instead of masquerading as a legitimate 0. Tag it
        # as a grading error (and zero the reward, which is not on the task's
        # scale and would otherwise skew mean reward); the raw sentinel is kept in
        # metadata and a ``verifier_crash`` signal for triage.
        if out.reward < 0:
            logger.warning("Verifier wrote crash sentinel %s to %s", out.reward, path)
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="VerifierCrashError",
                signals=[*out.signals, Signal(name="verifier_crash", value=1.0)],
                metadata={**out.metadata, "error": f"verifier wrote crash sentinel {out.reward} to {path}", "reward_sentinel": out.reward},
            )
        return out

    # A missing reward file means the verifier produced no verdict — a verifier/
    # infra failure, NOT a legitimate score of 0 (a correctly-failing verifier
    # writes reward 0). Surface it as a typed grading error (Harbor-aligned
    # names) so the engine promotes it to GRADING_ERROR and the untrustworthy
    # reward is filtered, distinguishable from an agent that genuinely scored 0.
    if saw_empty:
        logger.warning("Reward file present but empty at one of: %s", paths)
        return EvalOutput(reward=0.0, is_correct=False, error="RewardFileEmptyError", metadata={"error": "reward file present but empty"})
    logger.warning("No reward file found at any of: %s", paths)
    return EvalOutput(reward=0.0, is_correct=False, error="RewardFileNotFoundError", metadata={"error": "no reward file found"})


def _parse_reward_json(raw: str) -> EvalOutput:
    """Parse a JSON reward file into an EvalOutput.

    Supports both ``{"reward": 0.5}`` and Harbor-style ``{"rewards": {...}}``.

    Every other top-level scalar becomes a :class:`Signal`, so whatever
    fine-grained state the grader reported alongside its verdict — SWE-bench
    style ``f2p_passed``/``p2p_failed``/``apply_failed`` counts, a partial
    score, per-suite tallies — lands on the episode (and in the eval report's
    signal averages) instead of being collapsed into one number. Graders differ
    per benchmark, so nothing here is keyed to a specific field name.
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
    for key, val in data.items():
        if key not in _RESERVED_REWARD_KEYS and isinstance(val, bool | int | float):
            signals.append(Signal(name=key, value=float(val)))

    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=signals,
        metadata=data.get("metadata", {}),
    )
