from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SWE_MODAL_INTEGRATION") != "1",
    reason="set RUN_SWE_MODAL_INTEGRATION=1 to run Modal sandbox integration tests",
)


def _first_task(dataset_name: str) -> dict:
    from swe.prepare_data import prepare_dataset

    return prepare_dataset(dataset_name, include_golden_patch=True)[0]


def _assert_single_clean_agent_commit(env, working_dir: str) -> None:
    from swe.tasks.common import run_sandbox_with_retries

    count = run_sandbox_with_retries(env, "git rev-list --count HEAD", cwd=working_dir, timeout=30)
    assert count["output"].strip() == "1"

    status = run_sandbox_with_retries(env, "git status --short", cwd=working_dir, timeout=30)
    assert status["output"].strip() == ""


def test_modal_swesmith_agent_setup_hides_git_history():
    from swe.environment import create_env
    from swe.tasks.common import close_sandbox_env
    from swe.tasks.swesmith import setup_swesmith_agent_env

    task = _first_task("swe_smith_go")
    env = create_env(task, command_timeout=120, sandbox_timeout=900)
    try:
        setup_swesmith_agent_env(env, task, "modal-smith", print)
        _assert_single_clean_agent_commit(env, task["working_dir"])
    finally:
        close_sandbox_env(env)


def test_modal_rebench_v2_agent_setup_hides_git_history():
    from swe.environment import create_env
    from swe.tasks.common import close_sandbox_env
    from swe.tasks.swe_rebench_v2 import setup_swe_rebench_v2_agent_env

    task = _first_task("swe_rebench_v2_go")
    env = create_env(task, command_timeout=120, sandbox_timeout=900)
    try:
        setup_swe_rebench_v2_agent_env(env, task, "modal-rebench", print)
        _assert_single_clean_agent_commit(env, task["working_dir"])
    finally:
        close_sandbox_env(env)
