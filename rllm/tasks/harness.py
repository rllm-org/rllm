"""Harness registry: name → AgentFlow constructor.

Built-in harnesses (``ReActHarness``, ``ClaudeCodeHarness``, ...) are
:class:`rllm.sandbox.sandboxed_flow.SandboxedAgentFlow`
subclasses; they implement the standard ``AgentFlow`` protocol. This
module just maps user-facing names (``"react"``, ``"claude-code"``)
to their classes for the CLI's ``--agent`` flag.
"""

from __future__ import annotations

from typing import Any

_HARNESS_REGISTRY: dict[str, type] = {}


def register_harness(name: str, cls: type) -> None:
    """Register a harness class under a name."""
    _HARNESS_REGISTRY[name] = cls


def load_harness(name: str) -> Any:
    """Resolve a harness name to a fresh instance.

    Args:
        name: Registered name (``"react"``, ``"claude-code"``) or
            colon-separated import path (``"my_module:MyHarness"``).
    """
    if ":" in name:
        import importlib

        module_path, attr_name = name.rsplit(":", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, attr_name)
        return cls()

    _ensure_builtins_registered()

    if name not in _HARNESS_REGISTRY:
        available = ", ".join(sorted(_HARNESS_REGISTRY.keys()))
        raise KeyError(f"Unknown harness '{name}'. Available: {available}")
    return _HARNESS_REGISTRY[name]()


def list_harnesses() -> list[str]:
    """Return registered harness names."""
    _ensure_builtins_registered()
    return sorted(_HARNESS_REGISTRY.keys())


def is_harness_name(name: str) -> bool:
    """True if *name* refers to a registered harness or import-path harness."""
    if not name:
        return False
    if ":" in name:
        return True
    _ensure_builtins_registered()
    return name in _HARNESS_REGISTRY


_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    """Lazy-register built-in harnesses on first use."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from rllm.harnesses import bash, claude_code, react  # noqa: F401

    _BUILTINS_REGISTERED = True
