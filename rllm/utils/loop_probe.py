"""Lightweight asyncio event-loop lag probe."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid %s=%r; using %.3f", name, value, default)
        return default


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * pct)))
    return sorted_values[idx]


async def _probe_loop(label: str, sample_interval_s: float, report_interval_s: float, logger: logging.Logger) -> None:
    loop = asyncio.get_running_loop()
    delays: list[float] = []
    started = loop.time()
    report_at = started + report_interval_s
    expected_ticks = 0

    while True:
        expected = loop.time() + sample_interval_s
        await asyncio.sleep(sample_interval_s)
        now = loop.time()
        expected_ticks += 1
        delays.append(max(0.0, now - expected))

        if now < report_at:
            continue

        elapsed = max(now - started, 1e-9)
        expected_by_time = max(1, int(round(elapsed / sample_interval_s)))
        delays_sorted = sorted(delays)
        mean = sum(delays) / len(delays) if delays else 0.0
        print(
            "event_loop_probe "
            f"label={label} elapsed_s={elapsed:.1f} ticks={expected_ticks} "
            f"expected_ticks={expected_by_time} tick_ratio={expected_ticks / expected_by_time:.3f} "
            f"lag_ms_mean={mean * 1000.0:.3f} "
            f"lag_ms_p50={_percentile(delays_sorted, 0.50) * 1000.0:.3f} "
            f"lag_ms_p95={_percentile(delays_sorted, 0.95) * 1000.0:.3f} "
            f"lag_ms_max={(delays_sorted[-1] if delays_sorted else 0.0) * 1000.0:.3f}",
            flush=True,
        )
        delays.clear()
        started = now
        report_at = now + report_interval_s
        expected_ticks = 0


def start_loop_probe(label: str, logger: logging.Logger | None = None) -> asyncio.Task | None:
    """Start a loop-lag probe when ``RLLM_LOOP_PROBE`` is enabled."""
    if not (_env_flag("RLLM_LOOP_PROBE") or _env_flag(f"RLLM_LOOP_PROBE_{label.upper()}")):
        return None

    sample_interval_s = max(0.001, _env_float("RLLM_LOOP_PROBE_SAMPLE_INTERVAL_S", 0.1))
    report_interval_s = max(sample_interval_s, _env_float("RLLM_LOOP_PROBE_REPORT_INTERVAL_S", 60.0))
    probe_logger = logger or logging.getLogger(__name__)
    print(
        f"starting event_loop_probe label={label} sample_interval_s={sample_interval_s:.3f} report_interval_s={report_interval_s:.3f}",
        flush=True,
    )
    return asyncio.create_task(
        _probe_loop(label, sample_interval_s, report_interval_s, probe_logger),
        name=f"rllm-loop-probe-{label}",
    )


async def stop_loop_probe(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
