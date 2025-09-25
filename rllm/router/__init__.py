"""Routing utilities for rollout engines."""

# Expose Router when importing package if needed
try:
    from .router import Router  # noqa: F401
except Exception:
    # Allow import of package even if Router has extra deps during runtime
    Router = None  # type: ignore
