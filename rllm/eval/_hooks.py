"""Back-compat shim. Use :mod:`rllm.hooks` directly going forward.

The class moved from ``rllm.eval._hooks.EvalHooks`` to
``rllm.hooks.SandboxTaskHooks`` because it isn't eval-specific — training
sandboxed harnesses use it too. The eval-named alias is kept here for
existing imports.
"""

from rllm.hooks import EvalHooks, SandboxTaskHooks

__all__ = ["EvalHooks", "SandboxTaskHooks"]
