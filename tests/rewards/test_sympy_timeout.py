import time


def test_sympy_timeout(monkeypatch):
    # Import inside to ensure env var is read at runtime.
    import rllm.rewards.math_utils.utils as utils

    def slow_simplify(_):
        time.sleep(5)
        return 0

    monkeypatch.setattr(utils.sympy, "simplify", slow_simplify)

    # Very small timeout so the test is fast.
    monkeypatch.setenv("RLLM_SYMPY_TIMEOUT_S", "0.05")

    start = time.time()
    ok = utils.are_equal_under_sympy("1", "1")
    elapsed = time.time() - start

    assert ok is False
    assert elapsed < 1.0
