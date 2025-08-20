from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Iterable, Optional
from dotenv import load_dotenv, find_dotenv

try:
    from strands import Agent
    from strands.models.openai import OpenAIModel
except Exception:  # pragma: no cover - allow import-time flexibility
    Agent = None  # type: ignore
    OpenAIModel = None  # type: ignore


def _resolve_strands_model_from_env():
    """
    Resolve a Strands provider model from environment variables.

    Supports either Together or OpenAI-compatible endpoints:
      - TOGETHER_AI_API_KEY / TOGETHER_AI_MODEL_NAME
      - OPENAI_API_KEY / MODEL_NAME / OPENAI_BASE_URL
    """
    # Load .env before reading environment variables
    load_dotenv(find_dotenv())

    together = os.getenv("TOGETHER_AI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if together:
        client_args = {"api_key": together, "base_url": "https://api.together.xyz/v1"}
        model_id = os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    elif openai_key:
        client_args = {"api_key": openai_key}
        if base_url:
            client_args["base_url"] = base_url
        model_id = os.getenv("MODEL_NAME", "gpt-4o")
    else:
        raise ValueError("Missing API key: set TOGETHER_AI_API_KEY or OPENAI_API_KEY")

    if OpenAIModel is None:
        raise ImportError("strands OpenAIModel is not available. Please install strands.")

    # Instantiate Strands' OpenAI provider
    return OpenAIModel(client_args=client_args, model_id=model_id, params={"temperature": 0.0, "max_tokens": 512})


def _normalize_event(e: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Map Strands stream_async() events to rLLM's 4 minimal event types.

    Output types (matching StrandsWorkflow expectations):
      - {"type": "TextDelta", "text": str}
      - {"type": "ToolUse", "toolUseId": str, "name": str, "input": dict}
      - {"type": "ToolResult", "toolUseId": str, "content": Any, "status": str}
      - {"type": "Stop", "final_text": Optional[str], "result": dict}
    """
    cb = e.get("callback", e)

    def _extract_textish(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            if "text" in x:
                return _extract_textish(x["text"])
            if "delta" in x:
                return _extract_textish(x["delta"])
            if "data" in x:
                return _extract_textish(x["data"])
            if str(x.get("type", "")).endswith(".delta") and "text" in x:
                return _extract_textish(x["text"])
            return ""
        if isinstance(x, (list, tuple)):
            return "".join(_extract_textish(y) for y in x)
        return str(x)

    # Text: pick ONE source, priority delta > data > text
    textish = cb.get("delta") or cb.get("data") or cb.get("text")
    txt = _extract_textish(textish) if textish is not None else ""
    if txt:
        yield {"type": "TextDelta", "text": txt}

    # Tool use begins
    if "current_tool_use" in cb:
        tu = cb["current_tool_use"] or {}
        yield {
            "type": "ToolUse",
            "toolUseId": tu.get("toolUseId") or tu.get("id") or "",
            "name": tu.get("name"),
            "input": tu.get("input") or {},
        }

    # Tool results embedded in a message
    if "message" in cb:
        msg = cb["message"] or {}
        for block in (msg.get("content") or []):
            if isinstance(block, dict) and block.get("type") in ("tool_result", "toolResult"):
                yield {
                    "type": "ToolResult",
                    "toolUseId": block.get("toolUseId") or block.get("tool_use_id") or "",
                    "content": block.get("content"),
                    "status": block.get("status") or "success",
                }

    # Stop/final result
    if "result" in cb:
        res = cb["result"]
        final_text = None
        try:
            # Try to extract final assistant text if present
            if isinstance(res, dict):
                msg = res.get("message") or {}
                # content could be list of blocks; try to join text blocks
                blocks = msg.get("content") or []
                texts: list[str] = []
                for b in blocks:
                    if isinstance(b, dict) and b.get("type") in ("text", "output_text") and b.get("text"):
                        texts.append(b.get("text"))
                if texts:
                    final_text = "".join(texts)
        except Exception:
            pass
        yield {"type": "Stop", "result": res, "final_text": final_text}

    if "stop" in e:
        yield {"type": "Stop", "result": e["stop"], "final_text": None}


class StrandsSession:
    """Session wrapping a Strands Agent for streaming events."""

    def __init__(self, tools: Optional[list] = None, system_prompt: Optional[str] = None):
        if Agent is None:
            raise ImportError("strands Agent is not available. Please install strands.")
        model = _resolve_strands_model_from_env()
        self._agent = Agent(
            model=model,
            tools=tools or [],
            system_prompt=system_prompt,
            callback_handler=None,
        )

    async def stream(self, user_text: str) -> AsyncIterator[Dict[str, Any]]:
        async for e in self._agent.stream_async(user_text):
            for out in _normalize_event(e):
                yield out

    # Provide a step() alias for compatibility with existing workflow code
    async def step(self, user_text: str) -> AsyncIterator[Dict[str, Any]]:
        async for ev in self.stream(user_text):
            yield ev

    async def close(self) -> None:
        if hasattr(self._agent, "aclose"):
            await self._agent.aclose()


def make_session_factory(default_tools: Optional[list] = None, default_system_prompt: Optional[str] = None):
    """
    Returns an async factory compatible with StrandsWorkflow, which expects
    to await a callable returning a session and supports passing tools/system_prompt.
    """

    async def _factory(*, tools: Optional[list] = None, system_prompt: Optional[str] = None, **kw):
        resolved_tools = tools or default_tools
        resolved_system_prompt = system_prompt or default_system_prompt
        return StrandsSession(tools=resolved_tools, system_prompt=resolved_system_prompt)

    return _factory


