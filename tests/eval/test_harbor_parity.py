"""Harbor→rLLM Path-B parity fixes in the loader + resolution layer.

These guard the behaviors that align rLLM's native sandbox path with how
Harbor actually starts/configures a task environment:

* prebuilt ``docker_image`` tasks boot as-is (no Dockerfile RUN replay), while
  base-image (SWE-bench style) tasks keep replay on;
* Harbor size strings (``memory = '4G'``) become ``memory_mb`` ints that
  actually reach the Modal/Daytona resource kwargs.
"""

from __future__ import annotations

from pathlib import Path

from rllm.eval._resolution import _sandbox_resource_kwargs, _should_replay_dockerfile
from rllm.tasks.loader import _load_task_from_dir, _parse_size_to_mb
from rllm.types import Task

# ---------------------------------------------------------------------------
# _parse_size_to_mb
# ---------------------------------------------------------------------------


def test_parse_size_suffixes():
    assert _parse_size_to_mb("4G") == 4096
    assert _parse_size_to_mb("10G") == 10240
    assert _parse_size_to_mb("512M") == 512
    assert _parse_size_to_mb("2g") == 2048  # case-insensitive
    assert _parse_size_to_mb("1T") == 1024 * 1024
    assert _parse_size_to_mb("4GB") == 4096  # trailing B tolerated
    assert _parse_size_to_mb("4GiB") == 4096  # GiB treated as GB here


def test_parse_size_passthrough_and_bad_values():
    assert _parse_size_to_mb(2048) == 2048  # already MB
    assert _parse_size_to_mb("2048") == 2048  # bare number = MB
    assert _parse_size_to_mb(None) is None
    assert _parse_size_to_mb("") is None
    assert _parse_size_to_mb("not-a-size") is None
    assert _parse_size_to_mb(True) is None  # bool is not a size


# ---------------------------------------------------------------------------
# prebuilt docker_image disables replay; base-image keeps it
# ---------------------------------------------------------------------------


def _write_task(tmp_path: Path, task_toml: str, dockerfile: str) -> Task:
    (tmp_path / "environment").mkdir(parents=True, exist_ok=True)
    (tmp_path / "environment" / "Dockerfile").write_text(dockerfile)
    (tmp_path / "task.toml").write_text(task_toml)
    return _load_task_from_dir(tmp_path, dataset_dir=tmp_path)


def test_prebuilt_image_disables_replay(tmp_path):
    """A task that ships its own docker_image boots as-is (Harbor parity)."""
    task = _write_task(
        tmp_path,
        '[environment]\ndocker_image = "org/img:t"\n',
        "FROM python:3.13-slim\nRUN apt-get install -y tmux\nCOPY x.py /app/x.py\n",
    )
    assert _should_replay_dockerfile(task) is False
    # The explicit image wins; the Dockerfile FROM does not override it.
    assert task.metadata["environment"]["docker_image"] == "org/img:t"


def test_base_image_swebench_style_keeps_replay(tmp_path):
    """No docker_image: FROM is a base, RUN extras must replay (SWE-bench)."""
    task = _write_task(
        tmp_path,
        "[environment]\ncpus = 1\n",
        "FROM swebench/sweb.eval.x86_64.django:latest\nWORKDIR /testbed\nRUN curl -LsSf https://astral.sh/uv | sh\n",
    )
    assert _should_replay_dockerfile(task) is True
    assert task.metadata["environment"]["docker_image"] == "swebench/sweb.eval.x86_64.django:latest"
    assert task.metadata["workdir"] == "/testbed"


def test_explicit_replay_flag_is_respected(tmp_path):
    """An explicit replay_dockerfile is never overridden by the prebuilt default."""
    task = _write_task(
        tmp_path,
        '[environment]\ndocker_image = "org/img:t"\nreplay_dockerfile = true\n',
        "FROM base\nRUN echo hi\n",
    )
    assert _should_replay_dockerfile(task) is True


# ---------------------------------------------------------------------------
# size strings reach the resource kwargs
# ---------------------------------------------------------------------------


def test_size_strings_become_mb_in_metadata(tmp_path):
    task = _write_task(
        tmp_path,
        "[environment]\ncpus = 1\nmemory = '4G'\nstorage = '10G'\n",
        "FROM ubuntu:24.04\nRUN echo hi\n",
    )
    env = task.metadata["environment"]
    assert env["memory_mb"] == 4096
    assert env["storage_mb"] == 10240


def test_modal_resource_kwargs_apply_memory_from_string_form():
    """The real swebench/terminal task.toml form ('4G') must reach Modal."""
    task = Task(
        id="t",
        instruction="",
        metadata={"environment": {"cpus": 1, "memory_mb": 4096}},
        dataset_dir=Path("."),
    )
    kw = _sandbox_resource_kwargs(task, "modal")
    assert kw["cpu"] == 1.0
    assert kw["memory"] == 4096
    assert "timeout" in kw  # lifetime is always sized for modal


def test_daytona_resource_kwargs_convert_mb_to_gb():
    task = Task(
        id="t",
        instruction="",
        metadata={"environment": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}},
        dataset_dir=Path("."),
    )
    kw = _sandbox_resource_kwargs(task, "daytona")
    assert kw["cpu"] == 2
    assert kw["memory"] == 4  # 4096 MB → 4 GB
    assert kw["disk"] == 10  # 10240 MB → 10 GB


# ---------------------------------------------------------------------------
# provider-agnostic sandbox lifetime knob (RLLM_SANDBOX_TIMEOUT_S)
# ---------------------------------------------------------------------------


def _budget_task() -> Task:
    return Task(
        id="t",
        instruction="",
        metadata={
            "environment": {"cpus": 1, "memory_mb": 2048, "storage_mb": 10240},
            "agent_timeout": 900,
            "verifier_timeout": 900,
        },
        dataset_dir=Path("."),
    )


def test_lifetime_floor_sized_from_task_budget_when_no_override(monkeypatch):
    """No override → both backends size lifetime to agent+verifier+install+slack."""
    monkeypatch.delenv("RLLM_SANDBOX_TIMEOUT_S", raising=False)
    monkeypatch.delenv("RLLM_MODAL_SANDBOX_TIMEOUT_S", raising=False)
    monkeypatch.delenv("RLLM_HARNESS_INSTALL_TIMEOUT_S", raising=False)
    t = _budget_task()
    # 900 (agent) + 900 (verifier) + 600 (install default) + 600 (slack) = 3000s
    assert _sandbox_resource_kwargs(t, "modal")["timeout"] == 3000
    assert _sandbox_resource_kwargs(t, "daytona")["auto_stop_interval"] == 50  # 3000s → 50 min


def test_provider_agnostic_override_applies_to_both_backends(monkeypatch):
    """RLLM_SANDBOX_TIMEOUT_S (seconds) floors Modal (seconds) and Daytona (minutes)."""
    monkeypatch.delenv("RLLM_MODAL_SANDBOX_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RLLM_SANDBOX_TIMEOUT_S", "4800")
    t = _budget_task()
    assert _sandbox_resource_kwargs(t, "modal")["timeout"] == 4800
    assert _sandbox_resource_kwargs(t, "daytona")["auto_stop_interval"] == 80  # 4800s → 80 min


def test_legacy_modal_var_is_deprecated_alias(monkeypatch):
    """RLLM_MODAL_SANDBOX_TIMEOUT_S still works; canonical name takes precedence."""
    from rllm.env import sandbox_timeout_override_s

    monkeypatch.delenv("RLLM_SANDBOX_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RLLM_MODAL_SANDBOX_TIMEOUT_S", "3600")
    assert sandbox_timeout_override_s() == 3600
    monkeypatch.setenv("RLLM_SANDBOX_TIMEOUT_S", "5000")  # canonical wins
    assert sandbox_timeout_override_s() == 5000
