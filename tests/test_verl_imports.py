"""Test that verl import paths are compatible with verl 0.7.1+.

Regression test for https://github.com/rllm-org/rllm/issues/470
After verl 0.7.1 restructured module paths, several imports in the
fully_async module broke with ModuleNotFoundError.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_path,names",
    [
        (
            "rllm.experimental.fully_async.runner",
            ["AsyncAgentTrainer", "FullyAsyncTaskRunner"],
        ),
        (
            "rllm.experimental.fully_async.fully_async_trainer",
            ["FullyAsyncTrainer"],
        ),
        (
            "rllm.experimental.fully_async.inference_manager",
            ["InferenceManager"],
        ),
    ],
)
def test_fully_async_imports(module_path: str, names: list[str]) -> None:
    """Verify fully_async modules can be imported without ModuleNotFoundError."""
    mod = importlib.import_module(module_path)
    for name in names:
        assert hasattr(mod, name), f"{module_path} is missing attribute {name}"


@pytest.mark.parametrize(
    "module_path,name",
    [
        ("verl.experimental.separation.ray_trainer", "SeparateRayPPOTrainer"),
        ("verl.experimental.separation.utils", "create_resource_pool_manager"),
        ("verl.experimental.separation.utils", "create_role_worker_mapping"),
        ("verl.experimental.agent_loop", "AgentLoopManager"),
        ("verl.utils.net_utils", "get_free_port"),
    ],
)
def test_verl_module_paths_exist(module_path: str, name: str) -> None:
    """Verify the verl module paths used by rllm exist in the installed verl package."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, name), f"{module_path} is missing attribute {name}"
