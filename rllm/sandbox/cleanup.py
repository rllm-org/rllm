"""Process-wide cleanup for active sandboxes.

Cloud sandboxes (Modal, Daytona, e2b, …) meter by duration and
*persist* if the rllm eval process dies. Docker sandboxes can also leak
under SIGKILL. This module keeps a process-global registry of every live
:class:`~rllm.sandbox.protocol.Sandbox` and installs an ``atexit`` +
``SIGTERM`` handler that closes them all on the way out.

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
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.sandbox.protocol import Sandbox

logger = logging.getLogger(__name__)

# Module-global. We use a regular set + lock rather than ``WeakSet``
# because cloud-sandbox handles often hold network connections that
# Python won't garbage-collect promptly enough to release the underlying
# resource. Explicit deregistration on close() is the source of truth.
_active: set[Sandbox] = set()
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


def close_all() -> None:
    """Close every sandbox in the registry. Idempotent and exception-safe."""
    with _lock:
        sandboxes = list(_active)
        _active.clear()
    for sandbox in sandboxes:
        try:
            sandbox.close()
        except Exception:
            # Don't let one stuck sandbox block the others. Log and move on.
            logger.exception("Sandbox close failed during cleanup")


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
