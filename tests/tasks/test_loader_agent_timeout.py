"""An absent per-task agent_timeout must stay absent (no phantom 600s default),
so the harness defers to the operator's RLLM_HARNESS_RUN_TIMEOUT_S instead of
min()'ing it down to a hidden 600s cap.
"""

from rllm.tasks.loader import _merge_task_toml_metadata


def test_undeclared_agent_timeout_is_omitted(tmp_path):
    (tmp_path / "task.toml").write_text('[agent]\nuser = "root"\n')  # no timeout_sec
    merged = _merge_task_toml_metadata(tmp_path, {})
    assert "agent_timeout" not in merged  # no phantom 600.0


def test_declared_agent_timeout_is_preserved(tmp_path):
    (tmp_path / "task.toml").write_text("[agent]\ntimeout_sec = 1800\n")
    merged = _merge_task_toml_metadata(tmp_path, {})
    assert merged["agent_timeout"] == 1800


def test_base_agent_timeout_survives_when_toml_silent(tmp_path):
    (tmp_path / "task.toml").write_text('[agent]\nuser = "root"\n')
    merged = _merge_task_toml_metadata(tmp_path, {"agent_timeout": 1234})
    assert merged["agent_timeout"] == 1234
