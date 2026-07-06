"""Event-loop health monitor.

Mirrors the model gateway's ``_loop_health_monitor`` so the *trainer's* single
event loop is observable too. It periodically logs:

- ``lag_ms`` — how much longer than the sample interval a bare ``sleep`` took,
  i.e. how long the loop couldn't run a scheduled callback (backlog / blocking).
- ``thread_cpu`` — this thread's own CPU utilisation over the window, which
  disambiguates *why*: high lag + high thread_cpu = self-CPU bound (do less /
  offload); high lag + low thread_cpu = the thread is starved (GIL contention).
- optional caller-supplied gauges (e.g. in-flight rollouts).

Diagnostic only — no behavioural effect. Run it as a background task on the loop
you want to observe and cancel it on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)
# This is a purpose-built diagnostic; keep it visible even when the surrounding
# rllm.* hierarchy is raised to WARNING. Only affects this logger.
logger.setLevel(logging.INFO)


async def run_loop_health_monitor(
    label: str,
    *,
    sample_s: float = 0.5,
    report_s: float = 20.0,
    gauges: Callable[[], str] | None = None,
) -> None:
    """Log ``label`` loop health every ``report_s`` until cancelled.

    ``gauges``, if given, returns a short string appended to each log line
    (e.g. ``"inflight=193 pending=42"``); exceptions from it are ignored.
    """
    lags: list[float] = []
    window_start = time.monotonic()
    last_cpu = time.thread_time()
    next_report = window_start + report_s
    while True:
        t0 = time.monotonic()
        try:
            await asyncio.sleep(sample_s)
        except asyncio.CancelledError:
            return
        lags.append(max(0.0, (time.monotonic() - t0 - sample_s) * 1000.0))
        now = time.monotonic()
        if now >= next_report and lags:
            window = now - window_start
            cpu = time.thread_time()
            util = 100.0 * (cpu - last_cpu) / window if window > 0 else 0.0
            ordered = sorted(lags)
            p50 = ordered[len(ordered) // 2]
            p99 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
            extra = ""
            if gauges is not None:
                try:
                    g = gauges()
                    if g:
                        extra = " | " + g
                except Exception:  # noqa: BLE001 - a gauge must never break the monitor
                    pass
            logger.info(
                "%s loop health: lag_ms p50=%.0f p99=%.0f max=%.0f | thread_cpu=%.0f%% | window=%.0fs%s",
                label,
                p50,
                p99,
                ordered[-1],
                util,
                window,
                extra,
            )
            lags.clear()
            last_cpu = cpu
            window_start = now
            next_report = now + report_s
