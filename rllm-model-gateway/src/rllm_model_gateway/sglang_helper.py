"""SGLang helpers for the gateway's ``use_sglang`` mode.

In ``use_sglang`` mode the gateway routes every chat turn through SGLang's
native ``/generate`` API (token-in ``input_ids``, token-out via
``meta_info.output_token_logprobs``), tokenizing prompts client-side with the
model's HF chat template and parsing completions with SGLang's own
FunctionCallParser / ReasoningParser — so a rollout turn is byte-identical to
the agent-harness + SGLang-server deployment path.

This module holds the stateless / parser-construction pieces of that path. The
I/O orchestration (routing, HTTP, trace persistence, accumulator updates) stays
on ``ReverseProxy`` (see proxy.py: ``_handle_sglang_generate`` /
``_sglang_generate_streaming`` / ``_render_sglang_prompt``), which calls these.

SGLang is an OPTIONAL dependency: ``rllm-model-gateway`` does not require it
(vLLM users never install it). Every ``sglang`` import here is therefore
function-local — this module imports cleanly without sglang installed, and the
``sglang`` packages load only when ``use_sglang`` mode actually runs a parser.
Do not hoist these imports to module scope.
"""

import json
import logging
import time
import uuid
from typing import Any

from rllm_model_gateway.models import TraceRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content flattening (chat-template prep)
# ---------------------------------------------------------------------------


def flatten_block_content(content: Any) -> str:
    """Flatten OpenAI/Anthropic block-format content to a plain string.

    A turn's content may arrive as a list of blocks, e.g.
    ``[{"type": "text", "text": "..."}]`` (OpenAI) or ``{"text": ...}`` /
    ``{"content": ...}`` (Anthropic/tool-result shapes). The renderers package
    expects string content and silently drops list content, so we join the text
    parts. A plain string passes through unchanged.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict):
            if b.get("type") in {"text", "input_text", "output_text"} or "text" in b:
                parts.append(str(b.get("text") or ""))
            elif "content" in b:
                parts.append(flatten_block_content(b.get("content")))
    return "\n".join(p for p in parts if p)


def flatten_message_content(message: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *message* with block-format content flattened to
    a string (other fields — role, tool_calls, tool_call_id — untouched)."""
    if not isinstance(message, dict) or isinstance(message.get("content"), str):
        return message
    out = dict(message)
    out["content"] = flatten_block_content(message.get("content"))
    return out


def prepare_template_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one message for chat-template rendering, matching SGLang:
    flatten block content, coerce None content to "", and parse JSON-string
    tool-call arguments to dicts."""
    msg = flatten_message_content(message)
    if not isinstance(msg, dict):
        return msg
    if msg.get("content") is None:
        msg = {**msg, "content": ""}
    tool_calls = msg.get("tool_calls")
    if msg.get("role") == "assistant" and isinstance(tool_calls, list):
        new_calls = []
        changed = False
        for call in tool_calls:
            fn = isinstance(call, dict) and call.get("function")
            args = fn and fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed_args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    new_calls.append(call)
                    continue
                new_calls.append({**call, "function": {**fn, "arguments": parsed_args}})
                changed = True
            else:
                new_calls.append(call)
        if changed:
            msg = {**msg, "tool_calls": new_calls}
    return msg


def apply_chat_template(
    messages: list[dict[str, Any]],
    tools: Any,
    *,
    tokenizer: Any,
    auto_adds_specials: bool,
) -> list[int]:
    """Tokenize the full message list to token ids EXACTLY as SGLang's
    /v1/chat/completions would, so a rollout turn is byte-identical to the
    agent-harness + SGLang-server deployment path it will be used in.

    Client-side (the gateway holds the tokenizer): authoritative for the served
    model and reachable in the sgl-router topology, which exposes no
    messages→token-ids endpoint (its /v1/tokenize is text-only).

    Mirrors SGLang's ``ServingChat._apply_jinja_template`` step for step:
      1. flatten OpenAI block-format content (``[{"type":"text",...}]``) to a
         string — the HF template silently drops list content otherwise (image/
         non-text blocks become a text placeholder; multimodal is out of scope);
      2. coerce ``content: None`` -> ``""`` (template-safe);
      3. parse assistant ``tool_calls[].function.arguments`` from a JSON string
         to a dict — tool-use chat templates expect a mapping, and a raw JSON
         string would render double-escaped on some models (Transformers docs);
      4. render with ``tokenize=False`` then ``encode`` with
         ``add_special_tokens=False`` when the tokenizer auto-adds specials, so
         the template's own role/special tokens are not doubled (e.g. BOS).

    ``auto_adds_specials`` is the proxy's one-time probe of whether
    ``tokenizer.encode("")`` returns any tokens (see ReverseProxy.__init__).
    """
    prepared = [prepare_template_message(m) for m in messages]
    rendered = tokenizer.apply_chat_template(
        prepared,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    encode_kwargs = {"add_special_tokens": False} if auto_adds_specials else {}
    return list(tokenizer.encode(rendered, **encode_kwargs))


# ---------------------------------------------------------------------------
# SGLang parsers (tool calls + reasoning) — sglang imported lazily
# ---------------------------------------------------------------------------


def get_fc_parser(tools: Any, tool_call_parser: str | None, cache: dict[str, Any]) -> Any:
    """Build (and cache) an SGLang FunctionCallParser for the request's tools.

    Cached (in the caller-owned ``cache`` dict) by the tool-name set so repeated
    turns reuse the detector. Returns None if no parser is configured or tools
    are absent. ``sglang`` is imported here, lazily, so the module loads without
    it installed.
    """
    if not tool_call_parser or not tools:
        return None
    try:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
    except ImportError:
        logger.warning("sglang FunctionCallParser unavailable; tool calls returned as text", exc_info=True)
        return None
    key = ",".join(sorted((t.get("function") or {}).get("name", "") for t in tools))
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        sg_tools = [Tool(type="function", function=Function(**t["function"])) for t in tools]
        parser = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_call_parser)
    except Exception:
        logger.warning("Failed to build FunctionCallParser(%r)", tool_call_parser, exc_info=True)
        return None
    cache[key] = parser
    return parser


def parse_completion(
    completion_ids: list[int],
    tools: Any,
    *,
    tokenizer: Any,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    fc_cache: dict[str, Any],
) -> dict[str, Any]:
    """Parse /generate completion token ids into ``{content, reasoning,
    tool_calls}`` using SGLang's own parsers (same as /v1/chat/completions and
    slime's agent parsing).

    Steps (mirrors slime.agent.parsing.parse_model_output):
      1. decode token ids -> text (the gateway's tokenizer),
      2. optional reasoning split via SGLang ReasoningParser,
      3. tool-call extraction via SGLang FunctionCallParser -> remaining text +
         ToolCallItem(name, parameters: JSON-str).

    Best-effort: any failure degrades to plain-text content (the agent still
    gets the text; only structured tool_calls are lost). ``sglang`` is imported
    lazily inside this function.
    """
    result: dict[str, Any] = {"content": "", "reasoning": None, "tool_calls": []}
    if not completion_ids or tokenizer is None:
        return result
    try:
        text = tokenizer.decode(list(completion_ids), skip_special_tokens=False)
    except Exception:
        logger.warning("tokenizer.decode failed; returning empty reply", exc_info=True)
        return result

    # 1. reasoning split
    if reasoning_parser:
        try:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            r, b = ReasoningParser(model_type=reasoning_parser, stream_reasoning=False).parse_non_stream(text)
            result["reasoning"], text = (r or None), (b or "")
        except Exception:
            logger.warning("ReasoningParser failed; keeping raw text", exc_info=True)

    # 2. tool-call extraction
    parser = get_fc_parser(tools, tool_call_parser, fc_cache)
    if parser is not None:
        try:
            if parser.has_tool_call(text):
                text, calls = parser.parse_non_stream(text)
                result["tool_calls"] = [{"name": c.name or "tool", "arguments": c.parameters or "{}"} for c in calls]
        except Exception:
            logger.warning("FunctionCallParser failed; returning text-only reply", exc_info=True)

    result["content"] = (text or "").strip()
    return result


# ---------------------------------------------------------------------------
# /generate request + response shaping
# ---------------------------------------------------------------------------


def sglang_sampling_params(request_body: dict[str, Any]) -> dict[str, Any]:
    """Translate the OpenAI chat body's sampling fields to SGLang's schema."""
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "no_stop_trim": True,
    }
    # max tokens: accept the common OpenAI spellings
    for key in ("max_completion_tokens", "max_tokens", "max_new_tokens"):
        if request_body.get(key) is not None:
            sp["max_new_tokens"] = int(request_body[key])
            break
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if request_body.get(src) is not None:
            sp[dst] = request_body[src]
    if request_body.get("stop"):
        sp["stop"] = request_body["stop"]
    return sp


def parse_sglang_generate(
    gen: dict[str, Any],
    prompt_ids: list[int],
) -> tuple[list[int], list[int], list[float], str, str | None]:
    """Extract (prompt_ids, completion_ids, logprobs, text, finish_reason) from
    an SGLang /generate response dict.

    ``meta_info.output_token_logprobs`` is a list of ``[logprob, token_id, ...]``
    (the canonical token-id+logprob source — see slime's call_sglang_generate).
    SGLang does not echo the prompt ids, so we keep the ones we sent.
    """
    meta = gen.get("meta_info") or {}
    otl = meta.get("output_token_logprobs") or []
    completion_ids = [int(x[1]) for x in otl]
    logprobs = [float(x[0]) for x in otl]
    text = gen.get("text", "") or ""
    finish = (meta.get("finish_reason") or {}).get("type") if isinstance(meta.get("finish_reason"), dict) else None
    # Prefer engine-provided prompt ids if present, else the ids we sent.
    prompt_out = meta.get("prompt_token_ids") or list(prompt_ids)
    return list(prompt_out), completion_ids, logprobs, text, finish


def build_generate_trace(
    session_id: str,
    request_body: dict[str, Any],
    prompt_ids: list[int],
    completion_ids: list[int],
    logprobs: list[float],
    text: str,
    finish_reason: str | None,
    latency_ms: float,
    *,
    weight_version: int | None = None,
) -> TraceRecord:
    """Build a TraceRecord from native /generate token ids + logprobs."""
    return TraceRecord(
        trace_id=uuid.uuid4().hex,
        session_id=session_id,
        model=request_body.get("model", ""),
        messages=request_body.get("messages", []) or [],
        prompt_token_ids=list(prompt_ids),
        response_message={"role": "assistant", "content": text},
        completion_token_ids=list(completion_ids),
        logprobs=list(logprobs) if logprobs else None,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        token_counts={"prompt": len(prompt_ids), "completion": len(completion_ids)},
        weight_version=weight_version,
    )


def to_openai_tool_calls(parsed_tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert parsed tool calls -> OpenAI wire tool_calls.

    Input shape (from parse_completion / SGLang FunctionCallParser):
    ``{"name": str, "arguments": <JSON string>}``. OpenAI clients (incl. Strands)
    expect a unique ``id``, ``type": "function"``, ``function.name``, and
    ``function.arguments`` as a JSON-encoded **string** — so arguments pass
    through verbatim when already a string, else are JSON-encoded.
    """
    out: list[dict[str, Any]] = []
    for tc in parsed_tool_calls:
        name = tc.get("name", "") or ""
        args = tc.get("arguments", "{}")
        args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        out.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return out


def generate_to_chat_response(
    request_body: dict[str, Any],
    text: str,
    finish_reason: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    parsed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render an OpenAI chat.completion response so the agent sees a normal reply.

    Mirrors SGLang's native /v1/chat/completions exactly (serving_chat.py builds
    ``ChatMessage(content=text if text else "", tool_calls=..., reasoning_content=...)``):
    the parser's leftover ``normal_text`` is ALWAYS surfaced as ``content`` — never
    dropped — and ``tool_calls`` / ``reasoning_content`` are added alongside it when
    present. ``parsed`` (``{content, reasoning, tool_calls}``) already holds the
    post-parse fields from parse_completion; ``content`` is the text remaining after
    reasoning + tool-call extraction. Emitting all three lossless lets the agent send
    the same message list back next turn, which we then re-tokenize identically to
    how SGLang's chat API would. The gateway adds no discards of its own.
    """
    message: dict[str, Any] = {"role": "assistant"}
    wire_finish = finish_reason or "stop"

    parsed = parsed or {}
    tool_calls = parsed.get("tool_calls")
    content = parsed.get("content")
    reasoning = parsed.get("reasoning")

    # content is always present (empty string if the whole turn was tool call /
    # reasoning), matching SGLang native; tool_calls are added, not substituted.
    message["content"] = content if content is not None else (text or "")
    if tool_calls:
        message["tool_calls"] = to_openai_tool_calls(tool_calls)
        wire_finish = "tool_calls"
    if reasoning:
        message["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_body.get("model", ""),
        "choices": [{"index": 0, "message": message, "finish_reason": wire_finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
