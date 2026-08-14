"""Load the in-sandbox GDPval driver so its logic is unit-testable.

The driver runs inside the task sandbox against a Stirrup virtualenv that the
repo does not depend on. Stirrup is used when it happens to be importable, so
the real API is exercised; otherwise the handful of Stirrup symbols the driver
imports are stubbed. Either way the code under test is the driver source that
actually ships, not a copy.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from rllm.harnesses.stirrup import _DRIVER_SCRIPT

_STIRRUP_MODULES = (
    "stirrup",
    "stirrup.clients",
    "stirrup.clients.chat_completions_client",
    "stirrup.core",
    "stirrup.core.models",
    "stirrup.tools",
    "stirrup.tools.code_backends",
    "stirrup.tools.code_backends.base",
    "stirrup.tools.view_image",
    "stirrup.tools.web",
)


class _StubTool:
    """Stand-in for ``stirrup.core.models.Tool``, which is generic."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __class_getitem__(cls, _item):
        return cls


class _StubToolResult:
    def __init__(self, content=None, metadata=None, success=True):
        self.content = content
        self.metadata = metadata
        self.success = success


class _StubCodeExecToolProvider:
    def __init__(self, allowed_commands=None, shell_timeout=300):
        self._allowed_commands = allowed_commands
        self._shell_timeout = shell_timeout

    def _check_allowed(self, cmd):
        return True

    def get_code_exec_tool(self, *, name="code_exec", description=None):
        return _StubTool(name=name, description=description)

    def get_view_image_tool(self, *, name="view_image", description=None):
        return _StubTool(name=name, description=description)


def _stub_stirrup() -> dict[str, types.ModuleType]:
    """Build the minimum module tree the driver imports at module scope."""
    modules = {name: types.ModuleType(name) for name in _STIRRUP_MODULES}

    modules["stirrup"].Agent = object
    modules["stirrup"].aggregate_metadata = lambda metadata, **_kwargs: metadata
    modules["stirrup.clients.chat_completions_client"].ChatCompletionsClient = object

    core_models = modules["stirrup.core.models"]
    core_models.AssistantMessage = type("AssistantMessage", (), {})
    core_models.ImageContentBlock = type("ImageContentBlock", (), {"__init__": lambda self, data=None: setattr(self, "data", data)})
    core_models.Tool = _StubTool
    core_models.ToolResult = _StubToolResult
    core_models.ToolUseCountMetadata = type("ToolUseCountMetadata", (), {})

    base = modules["stirrup.tools.code_backends.base"]
    base.CodeExecToolProvider = _StubCodeExecToolProvider
    base.CodeExecutionParams = type("CodeExecutionParams", (), {})
    base.CommandResult = type(
        "CommandResult",
        (),
        {"__init__": lambda self, **kwargs: self.__dict__.update({"error_kind": None, "advice": None, **kwargs})},
    )

    modules["stirrup.tools.view_image"].ViewImageToolProvider = object
    modules["stirrup.tools.web"].WebToolProvider = object
    return modules


def load_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Execute the shipped driver source against a temporary sandbox layout."""
    workdir = tmp_path / "work"
    submission_dir = tmp_path / "submission"
    workdir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RLLM_STIRRUP_WORKDIR", str(workdir))
    monkeypatch.setenv("RLLM_STIRRUP_SUBMISSION_DIR", str(submission_dir))
    monkeypatch.setenv("RLLM_STIRRUP_RUN_METADATA_PATH", str(tmp_path / "run.json"))
    monkeypatch.setenv("RLLM_STIRRUP_SUBMITTABLE_ROOTS", json.dumps([str(workdir)]))
    monkeypatch.setenv("RLLM_STIRRUP_SHELL_TIMEOUT", "600")

    try:
        import stirrup  # noqa: F401

        stubs: dict[str, types.ModuleType] = {}
    except ImportError:
        stubs = _stub_stirrup()

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    module = types.ModuleType("gdpval_aa_driver")
    # Pydantic resolves a model's annotations through ``sys.modules[cls.__module__]``,
    # so the driver must be importable by name before its models are defined.
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(_DRIVER_SCRIPT, "driver.py", "exec"), module.__dict__)
    return module


@pytest.fixture
def driver(tmp_path, monkeypatch):
    return load_driver(tmp_path, monkeypatch)
