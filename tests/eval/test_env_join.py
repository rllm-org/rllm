"""Tests for the env-requirement join (``resolve_rollout_plan``) and the
wiring-time scan (``scan_env_requirements``) in :mod:`rllm.hooks`.

The invariant under test:

    needs_env = flow.needs_env ∨ evaluator-needs-env ∨ task-declares-env

plus the no-consumer rule (a task-declared env nobody can reach is downgraded
with a warning instead of provisioned for nobody).
"""

from __future__ import annotations

from pathlib import Path

import rllm
from rllm.hooks import FixedEvaluation, FromTaskEvaluation, flow_needs_env, resolve_rollout_plan, scan_env_requirements, task_declares_env
from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow
from rllm.types import Task

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@rllm.rollout(name="host-flow")
def _host_flow(task, config):
    return None


class _SandboxedStub(SandboxedAgentFlow):
    def run(self, task, config):
        return None


class _HostEvaluator:
    def evaluate(self, task, episode):
        return None


def _plain_task(tmp_path: Path, *, with_env_dir: bool = False, verifier_block: str = "") -> Task:
    bench = tmp_path / ("bench-env" if with_env_dir else "bench")
    bench.mkdir(exist_ok=True)
    (bench / "dataset.toml").write_text('[dataset]\nname = "bench"\n\n' + verifier_block)
    if with_env_dir:
        (bench / "environment").mkdir(exist_ok=True)
    return Task(id="t0", instruction="x?", metadata={}, dataset_dir=bench)


# ---------------------------------------------------------------------------
# Declared needs
# ---------------------------------------------------------------------------


def test_rollout_flows_declare_host_only():
    assert flow_needs_env(_host_flow) is False


def test_sandboxed_flows_declare_env():
    assert flow_needs_env(_SandboxedStub()) is True


def test_task_declares_env_via_environment_dir(tmp_path):
    assert task_declares_env(_plain_task(tmp_path, with_env_dir=True)) is True
    assert task_declares_env(_plain_task(tmp_path)) is False


def test_task_declares_env_via_task_path_metadata():
    assert task_declares_env({"task_path": "/some/harbor/task"}) is True
    assert task_declares_env({"question": "2+2?"}) is False


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


def test_host_flow_host_evaluator_no_env(tmp_path):
    """Case 1 shape: classic math RL — nothing requires a sandbox."""
    task = _plain_task(tmp_path)
    plan = resolve_rollout_plan(task, _host_flow, FixedEvaluation(_HostEvaluator()))
    assert plan.needs_env is False
    assert plan.verifier_kind == "fixed"


def test_sandboxed_flow_forces_env_even_with_fixed_evaluator(tmp_path):
    """Case 2 shape: the flow's declared need wins — a host-side evaluator
    must not disable the sandbox."""
    task = _plain_task(tmp_path)
    plan = resolve_rollout_plan(task, _SandboxedStub(), FixedEvaluation(_HostEvaluator()))
    assert plan.needs_env is True


def test_env_verifier_forces_env_for_host_flow(tmp_path):
    """Case 3 shape: a sandbox-shell verifier needs the env even when the flow doesn't."""
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "tests").mkdir()
    (bench / "tests" / "test.sh").write_text("#!/bin/sh\necho ok\n")
    (bench / "dataset.toml").write_text('[dataset]\nname = "bench"\n')
    task = Task(id="t0", instruction="x?", metadata={}, dataset_dir=bench)

    plan = resolve_rollout_plan(task, _host_flow, FromTaskEvaluation())
    assert plan.verifier_kind == "sandbox-shell"
    assert plan.needs_env is True


def test_no_consumer_rule_downgrades_task_env(tmp_path, caplog):
    """A task-declared env with a host-only flow and host-only verifier is
    downgraded (with a warning) instead of provisioned for nobody."""
    task = _plain_task(tmp_path, with_env_dir=True, verifier_block='[verifier]\nname = "math_reward_fn"\n')

    import rllm.hooks as hooks_mod

    hooks_mod._warned_no_consumer.discard(task.id)  # the warning fires once per task id
    with caplog.at_level("WARNING", logger="rllm.hooks"):
        plan = resolve_rollout_plan(task, _host_flow, FromTaskEvaluation())

    assert plan.verifier_kind == "registered"
    assert plan.needs_env is False
    assert any("skipping sandbox provisioning" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Wiring-time scan
# ---------------------------------------------------------------------------


def test_scan_flow_declaration_wins():
    scan = scan_env_requirements(_SandboxedStub(), None, None)
    assert scan.needs_env is True
    assert scan.any_remote is False


def test_scan_finds_env_row_beyond_the_first(tmp_path):
    """An env requirement may appear on any row, not just the first."""
    rows = [{"question": "2+2?"}, {"task_path": "/harbor/task-001"}]
    scan = scan_env_requirements(_host_flow, rows)
    assert scan.needs_env is True


def test_scan_detects_remote_backend_from_rows():
    rows = [{"task_path": "/t", "sandbox_backend": "daytona"}]
    scan = scan_env_requirements(_host_flow, rows)
    assert scan.any_remote is True


def test_scan_explicit_backend_overrides_row_metadata():
    """An explicit local backend wins over remote row metadata (same precedence
    as _resolve_backend at provision time) — no tunnel."""
    rows = [{"task_path": "/t", "sandbox_backend": "daytona"}]
    scan = scan_env_requirements(_host_flow, rows, sandbox_backend="docker")
    assert scan.needs_env is True
    assert scan.any_remote is False


def test_scan_explicit_remote_backend():
    scan = scan_env_requirements(_SandboxedStub(), None, sandbox_backend="daytona")
    assert scan.any_remote is True


def test_scan_no_env_anywhere():
    rows = [{"question": "2+2?", "ground_truth": "4"}]
    scan = scan_env_requirements(_host_flow, rows)
    assert scan.needs_env is False
    assert scan.any_remote is False
