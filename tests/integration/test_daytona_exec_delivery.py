"""Live regression tests: Daytona exec completions must be DELIVERED, not
silently dropped.

Background (2026-07-01): the SDK's one-shot ``process.exec`` rides a single
HTTP request that stays byte-silent until the command completes. NAT
middleboxes on the client path (measured: WSL2, drop threshold ~245-250s of
idle) silently discard the flow, black-holing the response — the command
finishes in the sandbox but exec() blocks until its read timeout and gets
mislabeled as a wall-clock timeout. rllm defends with (1) session-based exec
for long budgets and (2) TCP keepalive on the SDK pools.

Gating:
  DAYTONA_API_KEY        — required for all tests here (~40s sandbox roundtrips).
  RLLM_LIVE_SLOW=1       — additionally required for the >4-minute drop-line
                           tests, which prove delivery PAST the NAT idle
                           threshold and take ~5-6 minutes each.
"""

from __future__ import annotations

import os
import time

import pytest

requires_daytona = pytest.mark.skipif(
    not os.environ.get("DAYTONA_API_KEY"),
    reason="DAYTONA_API_KEY env var required",
)

requires_slow = pytest.mark.skipif(
    os.environ.get("RLLM_LIVE_SLOW") != "1",
    reason="RLLM_LIVE_SLOW=1 required (test idles >4min past the NAT drop threshold)",
)

# Past the measured drop line (~245-250s idle) with margin; a regression makes
# the exec block until its read timeout instead of returning at ~SLEEP_S.
DROP_LINE_SLEEP_S = 300


@pytest.fixture(scope="module")
def live_sandbox():
    """A raw SDK sandbox wrapped in DaytonaSandbox's exec (bypassing create)."""
    daytona = pytest.importorskip("daytona")
    from rllm.sandbox.backends.daytona import DaytonaSandbox, _enable_tcp_keepalive

    client = daytona.Daytona()
    _enable_tcp_keepalive(client)
    raw = client.create()
    sb = DaytonaSandbox.__new__(DaytonaSandbox)
    sb.name = raw.id[:8]
    sb._closed = False
    sb._persistent_env = {}
    sb._sandbox = raw
    try:
        yield sb
    finally:
        raw.delete()


@requires_daytona
def test_session_path_delivers_quickly(live_sandbox):
    """A long-budget exec rides the session path and still returns promptly
    when the command is short — delivery machinery works end to end."""
    t0 = time.monotonic()
    out = live_sandbox.exec("sleep 20; echo ALIVE", timeout=300)  # > threshold → session
    wall = time.monotonic() - t0
    assert "ALIVE" in out
    assert wall < 60, f"session exec took {wall:.0f}s for a 20s command"


@requires_daytona
@requires_slow
def test_session_path_delivers_past_nat_drop_line(live_sandbox):
    """THE regression: a command idling past the NAT drop threshold must
    return its output — with the old one-shot path this lost the response
    100% of the time (bisect 2026-07-01: 250/260/270/300/315/330s all lost)."""
    t0 = time.monotonic()
    out = live_sandbox.exec(f"sleep {DROP_LINE_SLEEP_S}; echo DELIVERED", timeout=600)
    wall = time.monotonic() - t0
    assert "DELIVERED" in out
    assert wall < DROP_LINE_SLEEP_S + 60, (
        f"exec returned only after {wall:.0f}s for a {DROP_LINE_SLEEP_S}s command — "
        "completion was lost and recovered by a timeout, not delivered"
    )


@requires_daytona
@requires_slow
def test_keepalive_keeps_one_shot_alive_past_drop_line(live_sandbox, monkeypatch):
    """Defense-in-depth layer: with TCP keepalive injected (module fixture),
    even the one-shot path survives past the drop line. Forces one-shot by
    raising the session threshold above the command budget."""
    import rllm.sandbox.backends.daytona as daytona_mod

    monkeypatch.setattr(daytona_mod, "_SESSION_EXEC_THRESHOLD_S", 10_000.0)
    t0 = time.monotonic()
    out = live_sandbox.exec(f"sleep {DROP_LINE_SLEEP_S}; echo KA_DELIVERED", timeout=600)
    wall = time.monotonic() - t0
    assert "KA_DELIVERED" in out
    assert wall < DROP_LINE_SLEEP_S + 60, (
        f"one-shot exec with keepalive returned only after {wall:.0f}s — "
        "keepalive did not keep the flow alive"
    )
