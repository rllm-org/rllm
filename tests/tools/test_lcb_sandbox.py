"""Regression tests for the LiveCodeBench sandbox's result collection."""

import threading

from rllm.tools.code_tools import lcb_tool


class _ProcessThatPostsNothing:
    """Stand-in for a child that exits without posting a result.

    What really happens when the executed code calls ``os._exit``, the
    child segfaults, or the OS OOM-kills it: ``join`` returns, the
    process is gone, and the result queue stays empty forever.
    """

    def __init__(self, target=None, args=(), **kwargs):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False

    def terminate(self):
        pass

    def kill(self):
        pass


def _call_with_deadline(fn, deadline):
    """Run ``fn`` on a worker thread; return ``(still_running, result)``."""
    box = {}
    worker = threading.Thread(target=lambda: box.__setitem__("result", fn()), daemon=True)
    worker.start()
    worker.join(deadline)
    return worker.is_alive(), box.get("result")


def test_lcb_sandbox_returns_when_child_posts_no_result(monkeypatch):
    monkeypatch.setattr(lcb_tool.multiprocessing, "Process", _ProcessThatPostsNothing)

    still_running, result = _call_with_deadline(lambda: lcb_tool.lcb_sandbox("1 + 1", timeout=1), 30.0)

    assert not still_running, "lcb_sandbox blocked forever waiting on an empty result queue"
    assert result == ("Timeout", "", "")
