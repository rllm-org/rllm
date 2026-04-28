"""Tests for :func:`rllm.eval.reward_fns._resolver.get_verifier_system_prompt`.

The resolver reads ``[verifier]`` from a task's ``dataset.toml`` /
``task.toml`` and returns the ``SYSTEM_PROMPT`` exported by the matching
reward_fn module — used by harnesses to inject output-format hints.
"""

from __future__ import annotations

from pathlib import Path

from rllm.eval.reward_fns._resolver import get_verifier_system_prompt
from rllm.types import Task


def _benchmark(tmp_path: Path, *, verifier_block: str = "") -> Path:
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "dataset.toml").write_text(f'[dataset]\nname = "bench"\n\n{verifier_block}')
    return bench


# ---------------------------------------------------------------------------
# [verifier].name → score_fn module's SYSTEM_PROMPT
# ---------------------------------------------------------------------------


def test_resolves_math_system_prompt(tmp_path):
    bench = _benchmark(tmp_path, verifier_block='[verifier]\nname = "math_reward_fn"\n')
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

    prompt = get_verifier_system_prompt(task)
    assert prompt is not None
    assert "boxed" in prompt.lower()


def test_resolves_mcq_system_prompt(tmp_path):
    bench = _benchmark(tmp_path, verifier_block='[verifier]\nname = "mcq_reward_fn"\n')
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

    prompt = get_verifier_system_prompt(task)
    assert prompt is not None


def test_resolves_bfcl_system_prompt(tmp_path):
    bench = _benchmark(tmp_path, verifier_block='[verifier]\nname = "bfcl_reward_fn"\n')
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

    prompt = get_verifier_system_prompt(task)
    assert prompt is not None


# ---------------------------------------------------------------------------
# [verifier].import_path → arbitrary module's SYSTEM_PROMPT
# ---------------------------------------------------------------------------


def test_resolves_via_import_path(tmp_path):
    bench = _benchmark(
        tmp_path,
        verifier_block='[verifier]\nimport_path = "rllm.eval.reward_fns.math:evaluate"\n',
    )
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

    prompt = get_verifier_system_prompt(task)
    assert prompt is not None
    assert "boxed" in prompt.lower()


def test_import_path_to_module_without_system_prompt_returns_none(tmp_path):
    bench = _benchmark(
        tmp_path,
        verifier_block='[verifier]\nimport_path = "json:loads"\n',
    )
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
    # ``json`` has no SYSTEM_PROMPT attribute → None
    assert get_verifier_system_prompt(task) is None


# ---------------------------------------------------------------------------
# Sandbox-shell / module verifiers don't expose SYSTEM_PROMPT
# ---------------------------------------------------------------------------


def test_shell_script_verifier_returns_none(tmp_path):
    bench = _benchmark(tmp_path, verifier_block='[verifier]\nscript = "tests/test.sh"\n')
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
    assert get_verifier_system_prompt(task) is None


def test_python_module_verifier_returns_none(tmp_path):
    bench = _benchmark(tmp_path, verifier_block='[verifier]\nmodule = "tests.evaluate"\n')
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
    assert get_verifier_system_prompt(task) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unknown_registered_name_returns_none(tmp_path):
    bench = _benchmark(tmp_path, verifier_block='[verifier]\nname = "nope_reward_fn"\n')
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
    assert get_verifier_system_prompt(task) is None


def test_no_dataset_toml_returns_none(tmp_path):
    bench = tmp_path / "no-config"
    bench.mkdir()
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
    assert get_verifier_system_prompt(task) is None


def test_no_verifier_block_returns_none(tmp_path):
    bench = _benchmark(tmp_path, verifier_block="")  # only [dataset] section
    task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
    assert get_verifier_system_prompt(task) is None


def test_per_task_toml_takes_precedence(tmp_path):
    """When ``sub_dir`` is set, ``task.toml`` overrides the shared dataset.toml."""
    bench = _benchmark(tmp_path, verifier_block='[verifier]\nname = "f1_reward_fn"\n')
    sub = bench / "task-001"
    sub.mkdir()
    (sub / "task.toml").write_text('[verifier]\nname = "math_reward_fn"\n')
    task = Task(
        id="0",
        instruction="",
        metadata={},
        dataset_dir=bench,
        sub_dir=Path("task-001"),
    )
    prompt = get_verifier_system_prompt(task)
    assert prompt is not None
    assert "boxed" in prompt.lower()  # math, not f1
