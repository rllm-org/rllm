"""Tests for the eval curation filter DSL."""

import pytest

from rllm.eval.filter_dsl import FilterError, compile_filter


def _ns(*, n=4, n_correct=2, avg=0.5, best=1.0, worst=0.0, pass_at=None):
    pass_at = pass_at or {}

    def _at(name, k):
        if name == "pass":
            return pass_at.get(k, 0.0)
        if name == "avg":
            return avg
        if name == "best":
            return best
        if name == "worst":
            return worst
        raise FilterError(name)

    return {"avg": avg, "best": best, "worst": worst, "solved": n_correct > 0, "n": n, "n_correct": n_correct, "_at": _at}


@pytest.mark.parametrize(
    "expr,ns,expected",
    [
        ("solved", _ns(n_correct=1), True),
        ("solved", _ns(n_correct=0), False),
        ("0 < avg < 1", _ns(avg=0.5), True),
        ("0 < avg < 1", _ns(avg=1.0), False),
        ("0 < avg < 1", _ns(avg=0.0), False),
        ("avg >= 0.5 and best == 1", _ns(avg=0.5, best=1.0), True),
        ("n_correct >= 3", _ns(n_correct=2), False),
        ("best == 1 and avg < 0.5", _ns(best=1.0, avg=0.25), True),
        ("not solved", _ns(n_correct=0), True),
        ("n_correct / n > 0.4", _ns(n=4, n_correct=2), True),
    ],
)
def test_evaluate(expr, ns, expected):
    assert compile_filter(expr).evaluate(ns) is expected


def test_at_token_pass_at_k():
    f = compile_filter("pass@4 >= 0.5")
    assert f.evaluate(_ns(pass_at={4: 0.75})) is True
    assert f.evaluate(_ns(pass_at={4: 0.25})) is False


def test_avg_at_k_is_avg():
    # avg@k is k-invariant — the rewrite maps it to avg regardless of k.
    assert compile_filter("avg@8 == avg@2").evaluate(_ns(avg=0.3)) is True


@pytest.mark.parametrize(
    "expr",
    [
        "",
        "   ",
        "import os",  # statement, not an expression
        "__import__('os')",  # disallowed call
        "open('x')",  # disallowed call
        "avg.real",  # attribute access
        "unknown_name > 1",  # unknown name
        "avg > 'x'",  # string literal
        "avg = 1",  # assignment / not an eval expr
    ],
)
def test_rejects_bad_expressions(expr):
    with pytest.raises(FilterError):
        compile_filter(expr)
