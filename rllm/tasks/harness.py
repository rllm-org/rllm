"""AgentHarness protocol: a pluggable agent harness for solving Tasks.

A harness drives an agent to complete a task in a sandbox. Two flavors:

- **Host-side**: harness runs in Python on the host, calls the LLM via the
  LiteLLM proxy, and uses the sandbox as a tool (e.g., :class:`ReActHarness`).

- **In-sandbox**: harness installs a CLI agent (Claude Code, Codex, OpenCode,
  ...) inside the sandbox and runs it pointed at the LiteLLM proxy.

Harnesses are registered by name and selected via the CLI ``--agent`` flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rllm.experimental.eval.types import AgentConfig
    from rllm.sandbox.protocol import Sandbox
    from rllm.tasks.task import Task
    from rllm.types import Trajectory


@runtime_checkable
class AgentHarness(Protocol):
    """A pluggable agent harness.

    Implementations:

    - :class:`rllm.tasks.harnesses.react.ReActHarness` — Python ReAct loop
    - :class:`rllm.tasks.harnesses.claude_code.ClaudeCodeHarness` — Claude Code in sandbox

    A harness is a long-lived object reused across tasks. Per-task state
    (sandbox, current task) is passed to ``run`` rather than stored on ``self``.
    """

    name: str
    """Stable identifier used for registration and CLI ``--agent`` lookup."""

    def setup(self, sandbox: Sandbox, config: AgentConfig) -> None:
        """Install/configure the harness inside the sandbox.

        Called once per sandbox, after ``Task.setup``. For host-side harnesses,
        this is typically a no-op.
        """
        ...

    def run(self, task: Task, sandbox: Sandbox, config: AgentConfig) -> Trajectory:
        """Drive the agent to complete the task.

        Args:
            task: The :class:`Task` instance.
            sandbox: The configured sandbox (already had ``Task.setup`` and
                ``self.setup`` called on it).
            config: ``AgentConfig`` with ``base_url``, ``model``, ``session_uid``.

        Returns:
            A :class:`Trajectory` describing the agent's interactions.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_HARNESS_REGISTRY: dict[str, type[AgentHarness]] = {}


def register_harness(name: str, cls: type[AgentHarness]) -> None:
    """Register a harness class under a name."""
    _HARNESS_REGISTRY[name] = cls


def load_harness(name: str) -> AgentHarness:
    """Resolve a harness by name (or ``module:Class`` import path).

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

    # Trigger lazy registration of built-ins on first lookup
    _ensure_builtins_registered()

    if name not in _HARNESS_REGISTRY:
        available = ", ".join(sorted(_HARNESS_REGISTRY.keys()))
        raise KeyError(f"Unknown harness '{name}'. Available: {available}")
    return _HARNESS_REGISTRY[name]()


def list_harnesses() -> list[str]:
    """Return the names of all registered harnesses."""
    _ensure_builtins_registered()
    return sorted(_HARNESS_REGISTRY.keys())


_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    """Lazy-register built-in harnesses on first use."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    # Import for side effect: each module calls register_harness on import
    from rllm.tasks.harnesses import claude_code, react  # noqa: F401

    _BUILTINS_REGISTERED = True
