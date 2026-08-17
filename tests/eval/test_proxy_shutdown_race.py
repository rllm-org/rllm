"""The proxy monitor must not report a death for a Ctrl+C teardown.

A terminal SIGINT reaches the whole foreground process group, so the LiteLLM
proxy child usually exits (cleanly, code 0) before the parent's interrupt
handling calls shutdown_proxy() and sets _shutting_down — which made the
monitor print its 🚨 PROXY DIED banner on every user interrupt.
"""

import logging
import threading
import time

from rllm.eval.proxy import EvalProxyManager


class _ExitedProc:
    """Stub for a proxy process that has already exited cleanly."""

    def wait(self):
        return 0


def _manager_with_stub() -> EvalProxyManager:
    mgr = EvalProxyManager.__new__(EvalProxyManager)  # skip real startup
    mgr._proxy_process = _ExitedProc()
    mgr._read_stderr_tail = lambda max_lines=60: ""
    return mgr


def _run_monitor(mgr) -> None:
    mgr._start_proxy_monitor()
    for t in threading.enumerate():
        if t.name == "rllm-proxy-monitor":
            t.join(timeout=10.0)


def test_interrupt_teardown_is_not_reported_as_death(caplog):
    """Flag set shortly AFTER the child exits — the Ctrl+C ordering."""
    mgr = _manager_with_stub()

    def late_shutdown():
        time.sleep(0.3)
        mgr._shutting_down = True

    threading.Thread(target=late_shutdown).start()
    with caplog.at_level(logging.ERROR, logger="rllm.eval.proxy"):
        _run_monitor(mgr)
    assert "PROXY DIED" not in caplog.text


def test_genuine_death_is_still_reported(caplog):
    """No teardown ever declared — the banner must still fire."""
    mgr = _manager_with_stub()
    with caplog.at_level(logging.ERROR, logger="rllm.eval.proxy"):
        _run_monitor(mgr)
    assert "PROXY DIED" in caplog.text
