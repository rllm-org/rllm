"""
Shared statistics tracking for fully async rollout system.

This module provides thread-safe counters for tracking in-flight requests
across different components (generation, tool calls, refine calls).

Usage:
    from rllm.experimental.fully_async.stats import global_stats
    
    # Track a request
    await global_stats.start_request("tool_call")
    try:
        result = await do_tool_call()
    finally:
        await global_stats.end_request("tool_call", success=True, latency=elapsed)
    
    # Get stats
    stats = global_stats.get_all_stats_sync()
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ComponentStats:
    """Statistics for a single component (e.g., tool_call, refine, generation)."""
    in_flight: int = 0
    total_started: int = 0
    total_completed: int = 0
    total_failed: int = 0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    total_latency: float = 0.0

    def to_dict(self) -> dict:
        avg_latency = self.total_latency / max(1, self.total_completed)
        return {
            "in_flight": self.in_flight,
            "total_started": self.total_started,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "avg_latency": avg_latency,
            "min_latency": self.min_latency if self.min_latency != float('inf') else 0,
            "max_latency": self.max_latency,
        }


class GlobalStats:
    """
    Thread-safe global statistics tracker for multiple components.
    
    Components:
    - "generation": LLM generation requests
    - "tool_call": Tool execution requests
    - "refine": Refine agent requests
    - "http": Raw HTTP requests (tracked separately in client)
    """
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._components: Dict[str, ComponentStats] = {}
        self._created_at = time.time()
    
    def _get_or_create(self, component: str) -> ComponentStats:
        """Get or create stats for a component (NOT thread-safe, call under lock)."""
        if component not in self._components:
            self._components[component] = ComponentStats()
        return self._components[component]
    
    async def start_request(self, component: str) -> int:
        """Record start of a request. Returns current in-flight count."""
        async with self._lock:
            stats = self._get_or_create(component)
            stats.in_flight += 1
            stats.total_started += 1
            return stats.in_flight
    
    async def end_request(self, component: str, success: bool, latency: float):
        """Record end of a request."""
        async with self._lock:
            stats = self._get_or_create(component)
            stats.in_flight -= 1
            if success:
                stats.total_completed += 1
                stats.total_latency += latency
                stats.min_latency = min(stats.min_latency, latency)
                stats.max_latency = max(stats.max_latency, latency)
            else:
                stats.total_failed += 1
    
    async def get_component_stats(self, component: str) -> dict:
        """Get stats for a specific component."""
        async with self._lock:
            if component not in self._components:
                return ComponentStats().to_dict()
            return self._components[component].to_dict()
    
    async def get_all_stats(self) -> Dict[str, dict]:
        """Get stats for all components."""
        async with self._lock:
            return {name: stats.to_dict() for name, stats in self._components.items()}
    
    def get_all_stats_sync(self) -> Dict[str, dict]:
        """Get stats without awaiting (may have slight race conditions)."""
        return {name: stats.to_dict() for name, stats in self._components.items()}
    
    def get_summary_sync(self) -> str:
        """Get a one-line summary of all components."""
        parts = []
        for name, stats in self._components.items():
            parts.append(f"{name}:{stats.in_flight}/{stats.total_completed}")
        return ", ".join(parts) if parts else "no stats"
    
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._created_at


# Global singleton instance
global_stats = GlobalStats()


# Convenience decorators for tracking
def track_async(component: str):
    """Decorator to track async function calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            await global_stats.start_request(component)
            success = False
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            finally:
                latency = time.time() - start_time
                await global_stats.end_request(component, success, latency)
        return wrapper
    return decorator


class TrackContext:
    """Context manager for tracking request lifecycle."""
    
    def __init__(self, component: str):
        self.component = component
        self.start_time = None
        self.success = False
    
    async def __aenter__(self):
        self.start_time = time.time()
        await global_stats.start_request(self.component)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        self.success = exc_type is None
        await global_stats.end_request(self.component, self.success, latency)
        return False  # Don't suppress exceptions
    
    def mark_success(self):
        """Manually mark as success even if there's no exception."""
        self.success = True