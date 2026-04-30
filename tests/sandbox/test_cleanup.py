"""Tests for the process-wide sandbox cleanup registry.

Cloud sandboxes leak (and cost money) if rllm eval dies between
sandbox creation and teardown. The cleanup registry installs an atexit
+ SIGTERM handler that closes everything still alive.
"""

from __future__ import annotations

from rllm.sandbox import cleanup


class _FakeSandbox:
    """Minimal protocol-shaped stand-in. Only ``close`` matters here."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _reset_registry() -> None:
    """Tests must not leak state into each other or into the real eval flow."""
    with cleanup._lock:
        cleanup._active.clear()


def test_register_then_close_all_calls_close_once():
    _reset_registry()
    sb = _FakeSandbox()
    cleanup.register(sb)

    cleanup.close_all()

    assert sb.closed == 1


def test_close_all_is_idempotent():
    _reset_registry()
    sb = _FakeSandbox()
    cleanup.register(sb)

    cleanup.close_all()
    cleanup.close_all()  # nothing left in registry

    assert sb.closed == 1  # not double-closed


def test_deregister_prevents_close_all_from_touching_sandbox():
    _reset_registry()
    sb = _FakeSandbox()
    cleanup.register(sb)
    cleanup.deregister(sb)

    cleanup.close_all()

    assert sb.closed == 0


def test_close_all_continues_when_one_sandbox_raises():
    """One stuck sandbox must not block the others — duration-billed cloud
    handles MUST get terminated, even if a sibling's close() throws."""
    _reset_registry()

    class _Boom(_FakeSandbox):
        def close(self) -> None:
            raise RuntimeError("network error")

    bad = _Boom()
    good = _FakeSandbox()
    cleanup.register(bad)
    cleanup.register(good)

    cleanup.close_all()  # should not raise

    assert good.closed == 1


def test_register_is_idempotent_for_same_sandbox():
    _reset_registry()
    sb = _FakeSandbox()
    cleanup.register(sb)
    cleanup.register(sb)

    cleanup.close_all()

    assert sb.closed == 1


def test_handlers_are_installed_on_first_register():
    """First register() should trigger atexit handler installation, not later."""
    _reset_registry()
    cleanup._handlers_installed = False  # reset module flag
    sb = _FakeSandbox()
    cleanup.register(sb)
    assert cleanup._handlers_installed is True
    _reset_registry()
