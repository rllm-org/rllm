# tools_safety.py
import functools, inspect, traceback, os

def wrap_tool(tool_callable, *, name: str | None = None, return_trace: bool = False):
    """Turn exceptions into an error payload; otherwise pass through untouched."""
    @functools.wraps(tool_callable)
    async def _safe(*args, **kwargs):
        try:
            out = tool_callable(*args, **kwargs)
            if inspect.isawaitable(out):
                out = await out
            return out
        except Exception as e:
            payload = {"status": "error", "message": str(e)[:800]}
            if return_trace or os.getenv("RLLM_TOOL_TRACE") == "1":
                payload["trace"] = "".join(traceback.format_exc())[-2000:]
            return payload
    # keep a readable name for logs/telemetry
    _safe.__name__ = name or getattr(tool_callable, "__name__", "tool")
    return _safe
