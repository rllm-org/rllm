"""Process-wide cleanup for active sandboxes and gateway resources.

Cloud sandboxes (Modal, Daytona, e2b, …) meter by duration and
*persist* if the rllm eval process dies. The gateway tunnel
(cloudflared subprocess) and uvicorn server are similar — they need to
come down on every exit path, not just the happy ``finally`` in the
CLI.

The module keeps two registries and installs an ``atexit`` + ``SIGTERM``
handler that drains both on the way out:

1. **Sandboxes** — registered at ``create_sandbox()`` time, deregistered
   on close. Closed FIRST so any in-flight LLM call from a sandbox dies
   before its target (the gateway) does.
2. **Late callbacks** — registered by :class:`EvalGatewayManager.start`,
   deregistered on its ``shutdown``. Run AFTER sandboxes, so the tunnel
   and uvicorn server outlive the sandboxes by a beat.

SIGKILL can't be caught — it's a fundamental OS limit. But it's the only
gap; everything else (clean exit, exception, ``Ctrl+C``, ``kill <pid>``)
flows through here.

Usage:

    from rllm.sandbox.cleanup import register, deregister

    sandbox = create_sandbox(...)
    register(sandbox)
    try:
        ...
    finally:
        sandbox.close()
        deregister(sandbox)

For non-sandbox resources (gateway, tunnel):

    from rllm.sandbox.cleanup import register_late_cleanup, deregister_late_cleanup

    register_late_cleanup("gateway-<id>", manager.shutdown)
    try:
        ...
    finally:
        manager.shutdown()
        deregister_late_cleanup("gateway-<id>")
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.sandbox.protocol import Sandbox

logger = logging.getLogger(__name__)

# Module-global. We use a regular set + lock rather than ``WeakSet``
# because cloud-sandbox handles often hold network connections that
# Python won't garbage-collect promptly enough to release the underlying
# resource. Explicit deregistration on close() is the source of truth.
_active: set[Sandbox] = set()
# Late callbacks keyed by name so callers can deregister on graceful
# shutdown without touching siblings. Ordered by insertion (Python 3.7+
# dict iteration order).
_late_callbacks: dict[str, Callable[[], None]] = {}
_lock = threading.Lock()
_handlers_installed = False


def register(sandbox: Sandbox) -> None:
    """Add *sandbox* to the active registry. Installs handlers on first call."""
    global _handlers_installed
    with _lock:
        _active.add(sandbox)
        if not _handlers_installed:
            _install_handlers()
            _handlers_installed = True


def deregister(sandbox: Sandbox) -> None:
    """Remove *sandbox* from the active registry. Idempotent."""
    with _lock:
        _active.discard(sandbox)


def register_late_cleanup(name: str, fn: Callable[[], None]) -> None:
    """Register a callback to run after sandboxes during process-death cleanup.

    Used by :class:`~rllm.eval.gateway.EvalGatewayManager` to register
    its ``shutdown`` (kills tunnel subprocess, then uvicorn thread) so
    SIGTERM doesn't leave either resource leaked.
    """
    global _handlers_installed
    with _lock:
        _late_callbacks[name] = fn
        if not _handlers_installed:
            _install_handlers()
            _handlers_installed = True


def deregister_late_cleanup(name: str) -> None:
    """Remove a late callback by *name*. Idempotent."""
    with _lock:
        _late_callbacks.pop(name, None)


def close_all() -> None:
    """Close every sandbox + run every late callback. Idempotent and exception-safe."""
    with _lock:
        sandboxes = list(_active)
        _active.clear()
        callbacks = list(_late_callbacks.items())
        _late_callbacks.clear()
    # Phase 1: sandboxes first. In-flight LLM calls from sandboxes will
    # fail when the gateway dies in phase 2; closing sandboxes promptly
    # releases compute meters before that happens.
    for sandbox in sandboxes:
        try:
            sandbox.close()
        except Exception:
            # Don't let one stuck sandbox block the others. Log and move on.
            logger.exception("Sandbox close failed during cleanup")
    # Phase 2: late callbacks (tunnel, gateway). Order = registration
    # order, so multiple gateways tear down in start order.
    for name, fn in callbacks:
        try:
            fn()
        except Exception:
            logger.exception("Late cleanup callback '%s' failed", name)


def _install_handlers() -> None:
    atexit.register(close_all)

    # SIGTERM is what `kill <pid>` and most container orchestrators send.
    # Chain to any pre-existing handler so we don't break callers who
    # already installed their own.
    prev_term = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(signum, frame):  # type: ignore[no-untyped-def]
        close_all()
        if callable(prev_term) and prev_term not in (signal.SIG_IGN, signal.SIG_DFL):
            prev_term(signum, frame)
        elif prev_term == signal.SIG_DFL:
            # Re-raise default behaviour: terminate.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        # signal.signal only works in the main thread. The atexit handler
        # still covers normal exit; this branch is a no-op for non-main
        # callers (e.g., tests running in a worker thread).
        logger.debug("Could not install SIGTERM handler — not running in main thread")
