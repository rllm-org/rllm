from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rllm.data import gdpval_aa as aa
from rllm.eval.agent_loader import load_agent
from rllm.harnesses.gdpval_stirrup import GdpvalStirrupHarness
from rllm.harnesses.stirrup import _CONFIG_DIR, _DRIVER_SCRIPT, STIRRUP_VERSION, StirrupHarness, _benchmark_name, _player_id, _submission_dir, _usage_metrics
from rllm.types import AgentConfig, Task


def _task(tmp_path: Path | None = None) -> Task:
    task = Task(
        id="gdpval-1",
        instruction=aa.render_aa_gdpval_task_prompt("Read the workbook and produce a report.", ["/home/user/source.xlsx"]),
        metadata={
            "workdir": "/home/user",
            "agent_user": "user",
            "reference_files": ["/home/user/source.xlsx"],
        },
    )
    if tmp_path is not None:
        task.dataset_dir = tmp_path
        task.sub_dir = None
        # The manifest names the benchmark that actually ran, read from the
        # dataset rather than hardcoded, so the fixture has to declare one.
        (tmp_path / "dataset.toml").write_text('[dataset]\nname = "gdpval"\n')
    return task


def _config_for(model: str, **sampling_params) -> AgentConfig:
    return AgentConfig(
        base_url="http://gateway/sessions/test/v1",
        model=model,
        session_uid="gdpval-1:0",
        sampling_params=sampling_params,
        metadata={"gateway_auth_token": "gateway-token"},
    )


def _config(**sampling_params) -> AgentConfig:
    return AgentConfig(
        base_url="http://gateway/sessions/test/v1",
        model="z-ai/glm-5.2",
        session_uid="test",
        sampling_params=sampling_params,
        metadata={"gateway_auth_token": "gateway-token"},
    )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_stirrup_is_registered_as_a_builtin_agent():
    assert isinstance(load_agent("stirrup"), StirrupHarness)
    assert isinstance(load_agent("gdpval-stirrup"), GdpvalStirrupHarness)


def test_harness_pins_the_stirrup_release():
    """0.2.0 is a floor, not a preference: earlier releases leak Stirrup's
    internal ToolCall fields into the request, which strict providers reject."""
    harness = GdpvalStirrupHarness()

    assert tuple(int(part) for part in STIRRUP_VERSION.split(".")) >= (0, 2, 0)
    assert f"stirrup=={STIRRUP_VERSION}" in harness.install_script()


def test_harness_limits_match_the_published_contract():
    harness = GdpvalStirrupHarness()

    assert harness.max_turns == aa.AA_MAX_TURNS == 250
    assert harness.shell_timeout == aa.AA_SHELL_TIMEOUT_SEC == 600
    assert harness.context_summarization_cutoff == aa.AA_CONTEXT_SUMMARIZATION_CUTOFF == 0.7


def test_env_carries_the_gateway_and_the_aa_runtime_contract(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-secret")

    env = GdpvalStirrupHarness().build_env(_task(), _config(reasoning_effort="xhigh"))

    assert env["OPENAI_BASE_URL"] == "http://gateway/sessions/test/v1"
    assert env["OPENAI_API_KEY"] == "gateway-token"
    assert env["RLLM_STIRRUP_MODEL"] == "z-ai/glm-5.2"
    assert env["RLLM_STIRRUP_REASONING_EFFORT"] == "xhigh"
    assert env["BRAVE_API_KEY"] == "brave-secret"
    assert env["RLLM_STIRRUP_WORKDIR"] == "/home/user"
    assert env["RLLM_STIRRUP_MAX_TURNS"] == "250"
    assert env["RLLM_STIRRUP_SHELL_TIMEOUT"] == "600"
    assert env["RLLM_STIRRUP_CONTEXT_CUTOFF"] == "0.7"
    assert json.loads(env["RLLM_STIRRUP_SUBMITTABLE_ROOTS"]) == ["/home/user", "/tmp"]


def test_view_image_can_be_withheld_from_a_text_only_model(monkeypatch):
    """AA exposes View Image only to vision-capable models; offering it to a
    text-only one kills the run when the provider rejects the image block."""
    monkeypatch.setenv("RLLM_STIRRUP_ENABLE_VISION", "0")
    import importlib

    from rllm.harnesses import gdpval_stirrup as gdpval_module
    from rllm.harnesses import stirrup as module

    reloaded = importlib.reload(module)
    # The subclass captured the pre-reload base, so it has to be rebuilt too.
    reloaded_gdpval = importlib.reload(gdpval_module)
    try:
        assert reloaded.StirrupHarness.enable_vision is False
        env = reloaded_gdpval.GdpvalStirrupHarness().build_env(_task(), _config())
        assert env["RLLM_STIRRUP_ENABLE_VISION"] == "0"
    finally:
        monkeypatch.delenv("RLLM_STIRRUP_ENABLE_VISION", raising=False)
        importlib.reload(module)
        importlib.reload(gdpval_module)


def test_output_budget_is_not_below_stirrups_own_default():
    """Too low an output cap is fatal, not a truncation: Stirrup raises
    OutputTokenLimitError and the trajectory dies mid-tool-call."""
    harness = GdpvalStirrupHarness()

    assert harness.max_output_tokens >= 64_000
    assert harness.max_output_tokens <= harness.max_context_tokens

    env = harness.build_env(_task(), _config())
    assert env["RLLM_STIRRUP_MAX_OUTPUT_TOKENS"] == str(harness.max_output_tokens)
    assert env["RLLM_STIRRUP_MAX_CONTEXT_TOKENS"] == str(harness.max_context_tokens)


def test_reasoning_effort_must_be_a_string():
    with pytest.raises(ValueError, match="must be a string"):
        GdpvalStirrupHarness().build_env(_task(), _config(reasoning_effort=3))


def test_invocation_runs_stirrup_in_the_existing_sandbox():
    command = GdpvalStirrupHarness().build_invocation("task", _task(), _config())

    assert command.startswith(f"/opt/stirrup-venv/bin/python {_CONFIG_DIR}/driver.py")
    assert "docker" not in command.lower()


def _driver_imports() -> set[str]:
    """Names the driver actually imports, ignoring prose that mentions them."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(_DRIVER_SCRIPT)):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_driver_does_not_use_stirrups_temp_dir_confined_backend():
    """LocalCodeExecToolProvider rejects the very paths AA's prompt mandates."""
    imported = _driver_imports()

    assert "LocalCodeExecToolProvider" not in imported
    assert "DockerCodeExecToolProvider" not in imported
    assert "E2BCodeExecToolProvider" not in imported
    assert "CodeExecToolProvider" in imported


def test_driver_exposes_no_user_interaction_tool():
    """AA's system prompt promises the model cannot interact with the user."""
    assert "USER_INPUT_TOOL" not in _driver_imports()
    assert "user_input" not in _DRIVER_SCRIPT


def test_driver_defines_the_aa_finish_contract(driver):
    assert driver.FINISH_TOOL.name == "finish"
    assert driver.ABANDON_TOOL.name == "abandon_task_finish"
    assert set(driver.FinishParams.model_fields) == {"summary", "paths"}
    assert set(driver.AbandonParams.model_fields) == {"reason"}
    assert "finish_tool=[FINISH_TOOL, ABANDON_TOOL]" in _DRIVER_SCRIPT


def test_driver_applies_the_published_shell_timeout(driver):
    assert driver.SHELL_TIMEOUT == aa.AA_SHELL_TIMEOUT_SEC == 600
    assert driver.SandboxCodeExecToolProvider(driver.WORKDIR, shell_timeout=driver.SHELL_TIMEOUT)._shell_timeout == 600


def test_driver_passes_the_system_prompt_and_turn_limit_explicitly():
    """These are constructor arguments, so assert on the call site."""
    assert 'system_prompt=Path(os.environ["RLLM_STIRRUP_SYSTEM_PROMPT_PATH"]).read_text' in _DRIVER_SCRIPT
    assert 'max_turns=int(os.environ.get("RLLM_STIRRUP_MAX_TURNS", "250"))' in _DRIVER_SCRIPT
    assert 'context_summarization_cutoff=float(os.environ.get("RLLM_STIRRUP_CONTEXT_CUTOFF", "0.7"))' in _DRIVER_SCRIPT


# ---------------------------------------------------------------------------
# Execution identity
# ---------------------------------------------------------------------------


class _RecordingSandbox:
    """Records exec calls so the solver's identity can be asserted."""

    def __init__(self, run_data: dict | None = None, manifest: dict | None = None, local_root: Path | None = None):
        self.execs: list[tuple[str, str | None]] = []
        self._run_data = run_data
        self._manifest = manifest
        self._local_root = local_root
        self.downloads: list[tuple[str, str]] = []

    def exec(self, command, timeout=None, user=None):
        del timeout
        self.execs.append((command, user))
        if command.startswith("cat ") and GdpvalStirrupHarness.run_metadata_path.fget(GdpvalStirrupHarness) in command:
            return json.dumps(self._run_data) if self._run_data is not None else ""
        return ""

    def download_dir(self, remote_path, local_path):
        self.downloads.append((remote_path, local_path))
        destination = Path(local_path)
        (destination / "files" / "home" / "user").mkdir(parents=True, exist_ok=True)
        report = destination / "files" / "home" / "user" / "report.docx"
        report.write_bytes(b"report bytes")
        written = [str(report)]
        if self._manifest is not None:
            manifest_path = destination / "manifest.json"
            manifest_path.write_text(json.dumps(self._manifest))
            written.append(str(manifest_path))
        return written


def _finished_run() -> dict:
    return {
        "termination": {"type": "finish", "summary": "done", "submitted_paths": ["/home/user/report.docx"]},
        "turns": 12,
        "metadata": {"token_usage": [{"input": 10, "answer": 3, "reasoning": 2}]},
    }


def _sandbox_manifest() -> dict:
    return {
        "termination": {"type": "finish", "summary": "done", "submitted_paths": ["/home/user/report.docx"]},
        "artifacts": [
            {
                "submitted_path": "/home/user/report.docx",
                "bundle_path": "files/home/user/report.docx",
                "sha256": "sandbox-hash",
                "size_bytes": 12,
            }
        ],
        "rejected_paths": [],
    }


def test_solver_runs_as_the_task_agent_user_and_config_is_written_as_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))
    sandbox = _RecordingSandbox(run_data=_finished_run(), manifest=_sandbox_manifest())

    GdpvalStirrupHarness().run(_task(tmp_path), _config(), env=sandbox)

    invocation_users = [user for command, user in sandbox.execs if f"/opt/stirrup-venv/bin/python {_CONFIG_DIR}/driver.py" in command]
    assert invocation_users == ["user"]

    # /opt is not writable by the solver, so config files are staged as root.
    config_writes = [user for command, user in sandbox.execs if f"cat > {_CONFIG_DIR}/" in command]
    assert config_writes and all(user is None for user in config_writes)


def test_system_prompt_is_written_into_the_sandbox_verbatim(tmp_path, monkeypatch):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))
    sandbox = _RecordingSandbox(run_data=_finished_run(), manifest=_sandbox_manifest())

    GdpvalStirrupHarness().run(_task(tmp_path), _config(), env=sandbox)

    write = next(command for command, _ in sandbox.execs if f"cat > {_CONFIG_DIR}/system_prompt.txt" in command)
    assert aa.AA_GDPVAL_SYSTEM_PROMPT in write

    instruction_write = next(command for command, _ in sandbox.execs if f"cat > {_CONFIG_DIR}/instruction.txt" in command)
    assert "- /home/user/source.xlsx" in instruction_write


# ---------------------------------------------------------------------------
# Submission manifest
# ---------------------------------------------------------------------------


def test_run_preserves_the_submission_and_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))
    provenance = {
        "task_id": "gdpval-1",
        "dataset_repo": "openai/gdpval",
        "dataset_revision": "abc123",
        "sandbox_image_digest": aa.AA_BASE_IMAGE_DIGEST,
        "sandbox_platform": "linux/amd64",
        "reference_files": [{"path": "/home/user/source.xlsx", "sha256": "ref-hash", "size_bytes": 15}],
    }
    (tmp_path / "gdpval_aa.json").write_text(json.dumps(provenance))
    sandbox = _RecordingSandbox(run_data=_finished_run(), manifest=_sandbox_manifest())

    task = _task(tmp_path)
    episode = GdpvalStirrupHarness().run(task, _config(), env=sandbox)

    remote, local = sandbox.downloads[0]
    assert remote == GdpvalStirrupHarness.submission_dir.fget(GdpvalStirrupHarness)
    # Keyed benchmark / player / run / task__attempt.
    assert Path(local).name == "gdpval-1__0"
    assert Path(local).parent.parent.name == _player_id("z-ai/glm-5.2", "stirrup")
    assert episode.metrics["turns"] == 12
    assert episode.metrics["total_tokens"] == 15

    manifest = json.loads(Path(episode.artifacts["submission_manifest"]).read_text())
    assert manifest["benchmark"] == "gdpval"
    assert manifest["graded"] is False
    assert manifest["solver_model"] == "z-ai/glm-5.2"
    assert manifest["run_id"] == "test"
    assert manifest["dataset_revision"] == "abc123"
    assert manifest["sandbox_image_digest"] == aa.AA_BASE_IMAGE_DIGEST
    assert manifest["stirrup_version"] == STIRRUP_VERSION
    assert manifest["system_prompt_sha256"] == aa.sha256_text(aa.AA_GDPVAL_SYSTEM_PROMPT)
    assert manifest["task_prompt_sha256"] == aa.sha256_text(task.instruction)
    assert manifest["reference_files"] == provenance["reference_files"]
    assert manifest["termination"]["type"] == "finish"
    assert manifest["metrics"]["turns"] == 12

    artifact = manifest["artifacts"][0]
    assert artifact["submitted_path"] == "/home/user/report.docx"
    assert Path(artifact["local_path"]).read_bytes() == b"report bytes"
    assert artifact["size_bytes"] == len(b"report bytes")
    # Hashed from the bytes that actually landed on the host, not copied
    # through from the sandbox-side manifest.
    assert artifact["sha256"] == "2138e20f2c6de32409659c519601a485e21fe056a5e3d6c019f075e4203ce608"
    assert artifact["sandbox_sha256"] == "sandbox-hash"


def test_manifest_never_reports_a_quality_score(tmp_path, monkeypatch):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))
    sandbox = _RecordingSandbox(run_data=_finished_run(), manifest=_sandbox_manifest())

    episode = GdpvalStirrupHarness().run(_task(tmp_path), _config(), env=sandbox)

    manifest = json.loads(Path(episode.artifacts["submission_manifest"]).read_text())
    serialized = json.dumps(manifest).lower()
    for forbidden in ["pairwise", "win_rate", "quality_score", "elo", "reward"]:
        assert forbidden not in serialized, forbidden


def test_abandoned_run_records_the_reason_and_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))
    run_data = {"termination": {"type": "abandon_task_finish", "reason": "input missing"}, "turns": 4, "metadata": {}}
    sandbox = _RecordingSandbox(run_data=run_data, manifest={"termination": run_data["termination"], "artifacts": [], "rejected_paths": []})

    episode = GdpvalStirrupHarness().run(_task(tmp_path), _config(), env=sandbox)

    manifest = json.loads(Path(episode.artifacts["submission_manifest"]).read_text())
    assert manifest["termination"] == {"type": "abandon_task_finish", "reason": "input missing"}
    assert manifest["artifacts"] == []
    assert episode.artifacts["submitted_paths"] == []


def test_a_run_that_produced_nothing_is_still_recorded(tmp_path, monkeypatch):
    """A crashed solver is a fact the corpus needs, not a missing file."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))

    class _DeadSandbox:
        def exec(self, command, timeout=None, user=None):
            del command, timeout, user
            return ""

        def download_dir(self, remote_path, local_path):
            raise RuntimeError("sandbox died")

    episode = GdpvalStirrupHarness().run(_task(tmp_path), _config(), env=_DeadSandbox())

    manifest = json.loads(Path(episode.artifacts["submission_manifest"]).read_text())
    assert manifest["termination"] == {"type": "unknown", "reason": "the solver produced no run metadata"}
    assert manifest["artifacts"] == []
    assert manifest["run_id"] == "test"
    assert episode.artifacts["submission_dir"] is None


def test_usage_metrics_report_turns_and_tokens_only():
    """Tokens, never cost.

    Vendor rates change without notice and differ per account, so pricing here
    would freeze a dated rate into every saved result and make it read as
    authoritative long after it stopped being true.
    """
    metrics = _usage_metrics(
        {"turns": 7, "metadata": {"token_usage": [{"input": 1_000_000, "answer": 100_000, "reasoning": 50_000}]}},
    )

    assert metrics == {
        "turns": 7,
        "input_tokens": 1_000_000,
        "answer_tokens": 100_000,
        "reasoning_tokens": 50_000,
        "output_tokens": 150_000,
        "total_tokens": 1_150_000,
    }


# ---------------------------------------------------------------------------
# Driver: submission validation and staging
# ---------------------------------------------------------------------------


def test_finish_accepts_absolute_paths_to_existing_files(driver, tmp_path):
    deliverable = tmp_path / "work" / "report.docx"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    deliverable.write_text("report")

    assert driver.path_problem(str(deliverable)) is None


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda tmp: "report.docx", "is not an absolute path"),
        (lambda tmp: "./report.docx", "is not an absolute path"),
        (lambda tmp: str(tmp / "work"), "is a directory, not a file"),
        (lambda tmp: str(tmp / "work" / "missing.pdf"), "does not exist"),
        (lambda tmp: "/etc/passwd", "resolves outside the writable roots"),
        (lambda tmp: "", "is empty"),
    ],
)
def test_finish_rejects_invalid_submissions(driver, tmp_path, factory, expected):
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)

    assert expected in driver.path_problem(factory(tmp_path))


def test_finish_rejects_a_symlink_that_escapes_the_writable_roots(driver, tmp_path):
    (tmp_path / "work").mkdir(parents=True, exist_ok=True)
    escape = tmp_path / "work" / "escape.docx"
    escape.symlink_to("/etc/hosts")

    assert "resolves outside the writable roots" in driver.path_problem(str(escape))


def test_staging_preserves_the_submitted_path_mapping(driver, tmp_path):
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "a").mkdir()
    (work / "b").mkdir()
    # Same basename in two directories: flattening would lose one of them.
    (work / "a" / "report.docx").write_text("first")
    (work / "b" / "report.docx").write_text("second")

    artifacts, rejected = driver.stage_submission([str(work / "a" / "report.docx"), str(work / "b" / "report.docx")])

    assert rejected == []
    assert len({entry["bundle_path"] for entry in artifacts}) == 2
    for entry in artifacts:
        staged = driver.SUBMISSION_DIR / entry["bundle_path"]
        assert staged.is_file()
        assert staged.read_text() == Path(entry["submitted_path"]).read_text()
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] == staged.stat().st_size


def test_staging_reports_rejected_paths_instead_of_dropping_them(driver, tmp_path):
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "report.docx").write_text("report")

    artifacts, rejected = driver.stage_submission([str(work / "report.docx"), "/etc/passwd", "relative.docx"])

    assert [entry["submitted_path"] for entry in artifacts] == [str(work / "report.docx")]
    assert {entry["path"] for entry in rejected} == {"/etc/passwd", "relative.docx"}


def test_a_second_model_does_not_overwrite_the_first(tmp_path, monkeypatch):
    """The corpus must accumulate competitors, not replace them.

    Keying on session_uid alone (``<task>:<attempt>``) put every model's files
    in one directory: running kimi after glm deleted glm's deliverables and
    left glm's episode JSON pointing at kimi's output.
    """
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))
    task = _task(tmp_path)

    glm = _submission_dir(task, _config(), "stirrup", "run-1")
    kimi = _submission_dir(task, _config_for("moonshotai/kimi-k3"), "stirrup", "run-1")
    assert glm != kimi

    # A different scaffold is a different competitor, too.
    assert _submission_dir(task, _config(), "terminus2", "run-1") != glm
    # And two runs of the same player stay separate rather than clobbering.
    assert _submission_dir(task, _config(), "stirrup", "run-2") != glm

    # Same player, same run, different task/attempt still separate.
    other = Task(id="gdpval-2", instruction="", metadata=dict(task.metadata))
    other.dataset_dir = tmp_path
    assert _submission_dir(other, _config(), "stirrup", "run-1") != glm


def test_player_label_separates_configurations_of_one_model(tmp_path, monkeypatch):
    """Without a label, a re-run pools into the original and the two settings
    can never be ranked against each other."""
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / "rllm-home"))
    task = _task(tmp_path)
    plain = _submission_dir(task, _config(), "stirrup", "run-1")

    monkeypatch.setenv("RLLM_ARENA_PLAYER_LABEL", "vision-off")
    labelled = _submission_dir(task, _config(), "stirrup", "run-1")

    assert plain != labelled
    assert "vision-off" in str(labelled)


def test_benchmark_name_prefers_the_declared_dataset_name(tmp_path):
    """A dataset materialized to /tmp/gdpval-smoke still files under gdpval."""
    (tmp_path / "dataset.toml").write_text('[dataset]\nname = "gdpval"\n')
    task = _task(tmp_path)

    assert _benchmark_name(task.dataset_dir) == "gdpval"
    assert "gdpval" in str(_submission_dir(task, _config(), "stirrup", "run-1"))
