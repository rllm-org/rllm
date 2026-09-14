"""Optional run-scoped lifecycle hooks for eval AgentFlows."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalRunContext:
    dataset_name: str
    agent_name: str
    model: str
    base_url: str
    task_count: int
    concurrency: int
    attempts: int
    sampling_params: dict[str, Any] = field(default_factory=dict)
    run_dir: Path | None = None
    tasks: tuple[Any, ...] = field(default_factory=tuple, repr=False)


async def call_eval_lifecycle(obj: object, method_name: str, *args) -> Any:
    """Call an optional sync or async lifecycle method without blocking the loop."""
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    if inspect.iscoroutinefunction(method):
        return await method(*args)
    result = await asyncio.to_thread(method, *args)
    if inspect.isawaitable(result):
        return await result
    return result


__all__ = ["EvalRunContext", "call_eval_lifecycle"]
