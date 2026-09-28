"""Versioned FinQA multi-table evaluation flow.

This is an experiment-local AgentFlow loaded through rLLM's public
``module:object`` extension point.  It deliberately leaves the upstream FinQA
flow and evaluator untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from finqa_tools import TOOL_FNS, TOOL_SPECS
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "finqa-multitable-visible-complete-v2"
MAX_TURNS = 50
MAX_TOOL_TURNS = 45
FINAL_ATTEMPTS = MAX_TURNS - MAX_TOOL_TURNS
DISCOVERY_MAX_COMPLETION_TOKENS = 2048
FINAL_MAX_COMPLETION_TOKENS = 8192
MAX_TOOL_OUTPUT_CHARS = 8000
# A deterministic safety threshold that reserves room in the 49,152-token
# serving context for an 8,192-token final response.  It is deliberately based
# only on policy-visible messages and never on hidden task information.
FINALIZE_AT_TRANSCRIPT_CHARS = 90_000
LLM_TIMEOUT_SECONDS = 300
LLM_RETRIES_PER_TURN = 2

PROMPT_PATH = Path(__file__).parent / "prompts" / "multitable_v2_system_prompt.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
PROMPT_SHA256 = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()


def build_policy_visible_initial_messages(metadata: dict[str, Any], instruction: Any, task_id: str) -> list[dict]:
    """Build task-derived policy input through an explicit public allowlist."""
    qtype = str(metadata.get("question_type") or "").lower()
    if not qtype.startswith("multi_table"):
        raise ValueError(
            f"{PROTOCOL_VERSION} only accepts multi_table tasks; task={task_id!r} question_type={qtype!r}"
        )
    question = str(metadata.get("question") or instruction or "")
    if not question.strip():
        raise ValueError(f"{PROTOCOL_VERSION} received an empty public question for task={task_id!r}")
    company = str(metadata.get("company") or "").strip()
    if not company:
        raise ValueError(f"{PROTOCOL_VERSION} requires the public canonical company identifier for task={task_id!r}")
    policy_question = (
        f"Canonical tool company identifier (public dataset metadata): `{company}`\n"
        "Use this exact identifier in all company_name tool arguments.\n\n"
        f"{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": policy_question},
    ]


def _msg_to_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg
    data: dict[str, Any] = {"role": msg.role}
    if msg.content:
        data["content"] = msg.content
    if getattr(msg, "tool_calls", None):
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return data


def _truncate(value: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    removed = len(value) - limit
    return value[: limit // 2] + f"\n...(truncated {removed} chars)...\n" + value[-limit // 2 :]


def _exec_tool_call(tc: Any, accessed_tables: list[str]) -> tuple[str, bool]:
    """Execute one call and return ``(policy_visible_result, had_error)``."""
    name = tc.function.name
    fn = TOOL_FNS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'. Valid tools: {sorted(TOOL_FNS)}", True

    try:
        args = json.loads(tc.function.arguments or "{}")
    except (json.JSONDecodeError, TypeError) as exc:
        return f"Error: failed to parse tool arguments for {name}: {exc}", True
    if not isinstance(args, dict):
        return f"Error: arguments for {name} must be a JSON object, got {type(args).__name__}.", True

    if name == "get_table_info":
        table = args.get("table_name")
        if isinstance(table, str) and table.strip():
            accessed_tables.append(table.lower().strip())

    try:
        result = fn(**args)
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}", True
    except Exception as exc:  # noqa: BLE001 - tool errors must remain visible to policy
        return f"Error: {name} raised {type(exc).__name__}: {exc}", True
    text = _truncate(str(result))
    return text, text.lstrip().lower().startswith("error")


def _transcript_chars(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _contains_unparsed_tool_call(content: str) -> bool:
    """Detect tool syntax that vLLM returned as text after parser failure."""
    lowered = content.lower()
    return "<tool_call>" in lowered or "</tool_call>" in lowered


async def _create_with_retry(client: AsyncOpenAI, **kwargs):
    errors: list[str] = []
    for attempt in range(1, LLM_RETRIES_PER_TURN + 1):
        try:
            return await client.chat.completions.create(**kwargs), errors
        except Exception as exc:  # noqa: BLE001 - transient serving failures are retryable
            errors.append(f"attempt={attempt}:{type(exc).__name__}:{exc}")
    return None, errors


def _step(messages: list[dict], content: str) -> Step:
    return Step(
        chat_completions=list(messages),
        model_response=content,
        action=content,
        thought=content,
    )


@rllm.rollout(name="finqa-multitable-v2")
async def finqa_multitable_v2(task: Task, config: AgentConfig) -> Episode:
    meta = task.metadata or {}
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    messages = build_policy_visible_initial_messages(meta, task.instruction, str(task.id))
    accessed_tables: list[str] = []
    steps: list[Step] = []
    llm_errors: list[str] = []
    tool_error_count = 0
    tool_call_count = 0
    malformed_tool_call_count = 0
    final_response = ""
    last_nonempty_content = ""
    finalize_reason = "tool_turn_budget"

    for tool_turn in range(MAX_TOOL_TURNS):
        if _transcript_chars(messages) >= FINALIZE_AT_TRANSCRIPT_CHARS:
            finalize_reason = "context_reserve"
            break

        response, errors = await _create_with_retry(
            client,
            model=config.model,
            messages=messages,
            tools=TOOL_SPECS,
            max_completion_tokens=DISCOVERY_MAX_COMPLETION_TOKENS,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        llm_errors.extend(errors)
        if response is None:
            finalize_reason = "tool_phase_llm_error"
            break

        msg = response.choices[0].message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []
        messages.append(_msg_to_dict(msg))
        steps.append(_step(messages, content))
        if content.strip():
            last_nonempty_content = content

        if not tool_calls and _contains_unparsed_tool_call(content):
            # Hermes logs malformed JSON and returns the raw tool syntax as
            # ordinary assistant text.  Treating that text as a final answer
            # silently shortens the rollout and confounds policy quality with
            # parser recovery.  Keep the failure policy-visible and give the
            # model another ordinary tool-use turn.
            malformed_tool_call_count += 1
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Your previous tool call was not valid JSON and could not be executed. "
                        "Retry it as one native function call with a JSON object for arguments. "
                        "Do not emit literal <tool_call> tags."
                    ),
                }
            )
            continue

        if not tool_calls:
            final_response = content
            finalize_reason = "model_final"
            break

        for tc in tool_calls:
            output, had_error = _exec_tool_call(tc, accessed_tables)
            tool_call_count += 1
            tool_error_count += int(had_error)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        remaining = MAX_TOOL_TURNS - tool_turn - 1
        if remaining in {30, 20, 10, 5, 2, 1}:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Budget checkpoint: {remaining} tool-use model turns remain before mandatory final synthesis. "
                        "Prioritize unresolved template fields and reserve enough context to produce the complete answer."
                    ),
                }
            )

    if not final_response.strip():
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tool use is now closed. Using only the evidence already visible in this conversation, produce the "
                    "complete final answer now. Begin with `FINAL ANSWER:`. Preserve the requested template, fill every "
                    "supported field, include all requested sections, and make no tool calls."
                ),
            }
        )
        for _ in range(FINAL_ATTEMPTS):
            response, errors = await _create_with_retry(
                client,
                model=config.model,
                messages=messages,
                max_completion_tokens=FINAL_MAX_COMPLETION_TOKENS,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            llm_errors.extend(errors)
            if response is None:
                continue
            msg = response.choices[0].message
            content = msg.content or ""
            messages.append(_msg_to_dict(msg))
            steps.append(_step(messages, content))
            if content.strip():
                final_response = content
                last_nonempty_content = content
                break
            messages.append(
                {
                    "role": "system",
                    "content": "The previous final response was empty. Return the complete `FINAL ANSWER:` text now.",
                }
            )

    used_fallback = False
    if not final_response.strip() and last_nonempty_content.strip():
        final_response = last_nonempty_content
        used_fallback = True

    return Episode(
        trajectories=[Trajectory(name="finqa-multitable-v2", steps=steps)],
        artifacts={
            "answer": final_response,
            "accessed_tables": accessed_tables,
            "turns": len(steps),
            "tool_calls": tool_call_count,
            "tool_errors": tool_error_count,
            "malformed_tool_calls": malformed_tool_call_count,
            "llm_errors": llm_errors,
            "protocol_version": PROTOCOL_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "max_turns": MAX_TURNS,
            "max_tool_turns": MAX_TOOL_TURNS,
            "final_max_completion_tokens": FINAL_MAX_COMPLETION_TOKENS,
            "finalize_reason": finalize_reason,
            "final_fallback_used": used_fallback,
            "transcript_chars": _transcript_chars(messages),
        },
    )
