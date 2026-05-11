"""Canonical realistic multi-turn message chain fixtures (Qwen-family).

These fixtures exercise the moves a real ``math_tool_agent``-style workflow
makes: a system prompt + task, a thinking assistant that issues a tool call,
a tool response, more thinking, optionally another tool call/response, and
a final assistant answer.

Two variants are exposed because real engine inputs always end on a user or
tool message — the workflow asks the engine for the *next* assistant turn,
so by definition the prefix it submits never ends on assistant:

* ``CHAIN_ENDING_ON_TOOL`` — one tool round-trip followed by another query.
  Useful for property-(iv) extension tests.
* ``CHAIN_ENDING_ON_USER`` — the user sends a follow-up question after the
  assistant's final answer. Useful for confirming the user-after-assistant
  transition also preserves the extension property.

Each chain is a list of TYPED step dicts: ``{"label": str, "messages": list[dict]}``.
Steps are *append-only* — the messages of step k+1 are step k's messages plus
one or two new ones. Tests use this to assert
``tokens(parse(step_{k+1})) startswith tokens(parse(step_k))``.

The single-message-per-step granularity is deliberate: most workflow
transitions add exactly one message (the just-arrived user/tool), and the
fixture mirrors that. The transitions that add two at once (e.g. asst +
tool, when the asst was generated and the tool returned within the same
step) are also represented — but the input to the engine for the *next*
turn is the union, so the test only checks the union prefixes.

Message format follows ``rllm.parser.messages``:
* assistants carry an optional ``reasoning`` string (rLLM-canonical name).
* tool calls follow the OpenAI wire format
  ``{"id": ..., "type": "function", "function": {"name": ..., "arguments": json_str}}``.
"""

from __future__ import annotations

from typing import TypedDict


class ChainStep(TypedDict):
    label: str
    messages: list[dict]


SYSTEM_PROMPT = "You are Qwen, a helpful assistant. Use the provided tools to solve the user's math problems. Show your reasoning."

USER_QUESTION = "What is 12 * 7 + 3?"
FOLLOWUP_QUESTION = "Now what about 12 * 7 - 3?"

TOOL_CALL_1 = {
    "id": "call_compute_1",
    "type": "function",
    "function": {
        "name": "compute",
        "arguments": '{"expression": "12 * 7 + 3"}',
    },
}

TOOL_CALL_2 = {
    "id": "call_compute_2",
    "type": "function",
    "function": {
        "name": "compute",
        "arguments": '{"expression": "12 * 7 - 3"}',
    },
}

ASSISTANT_1_REASONING = "The user asks for 12*7+3. I should call the compute tool to be safe instead of computing in my head."

ASSISTANT_2_REASONING = "The tool returned 87. That answers the original question, so I will present the result without another tool call."

ASSISTANT_3_REASONING = "The follow-up changes the sign of 3. I will call compute again with the updated expression."


def _system() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(
    *,
    content: str = "",
    reasoning: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(tool_call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
    }


# ── Variant 1: chain ending on a tool message ──────────────────────────────
#
# Step progression (each step adds one or more messages):
#   S0: system + user
#   S1: + asst_with_tool_call_1   (after engine call 1)
#   S2: + tool_response_1          (after tool executor; END HERE for tests
#                                   that need a tool-ending input)
#
# The realistic engine-input transitions a workflow exercises are:
#   S0 -> S2 (engine sees [sys, user, asst, tool] going into turn 2)
# and at finer-grained inspection:
#   S0 -> S1 (only if you query the engine again before the tool returns;
#             not realistic but useful for completer state debugging)
#   S1 -> S2 (degenerate: same query side)

ASSISTANT_1 = _assistant(
    content="I'll use the compute tool to evaluate this safely.",
    reasoning=ASSISTANT_1_REASONING,
    tool_calls=[TOOL_CALL_1],
)

TOOL_RESPONSE_1 = _tool(
    tool_call_id="call_compute_1",
    name="compute",
    content="87",
)

CHAIN_ENDING_ON_TOOL: list[ChainStep] = [
    {
        "label": "S0_system_plus_user",
        "messages": [_system(), _user(USER_QUESTION)],
    },
    {
        "label": "S1_after_asst_tool_call",
        "messages": [_system(), _user(USER_QUESTION), ASSISTANT_1],
    },
    {
        "label": "S2_after_tool_response",
        "messages": [
            _system(),
            _user(USER_QUESTION),
            ASSISTANT_1,
            TOOL_RESPONSE_1,
        ],
    },
]


# ── Variant 2: chain ending on a follow-up user message ────────────────────
#
# Step progression:
#   S0: system + user
#   S1: + asst_with_tool_call_1
#   S2: + tool_response_1
#   S3: + asst_final (the model answers the original question)
#   S4: + followup_user (user sends another question; END HERE)
#
# Realistic engine-input transitions:
#   S0 -> S2 -> S4   (every engine call sees a prefix ending in user/tool)

ASSISTANT_2_FINAL = _assistant(
    content="12 * 7 + 3 = 87.",
    reasoning=ASSISTANT_2_REASONING,
)

CHAIN_ENDING_ON_USER: list[ChainStep] = [
    {
        "label": "S0_system_plus_user",
        "messages": [_system(), _user(USER_QUESTION)],
    },
    {
        "label": "S1_after_asst_tool_call",
        "messages": [_system(), _user(USER_QUESTION), ASSISTANT_1],
    },
    {
        "label": "S2_after_tool_response",
        "messages": [
            _system(),
            _user(USER_QUESTION),
            ASSISTANT_1,
            TOOL_RESPONSE_1,
        ],
    },
    {
        "label": "S3_after_asst_final",
        "messages": [
            _system(),
            _user(USER_QUESTION),
            ASSISTANT_1,
            TOOL_RESPONSE_1,
            ASSISTANT_2_FINAL,
        ],
    },
    {
        "label": "S4_after_followup_user",
        "messages": [
            _system(),
            _user(USER_QUESTION),
            ASSISTANT_1,
            TOOL_RESPONSE_1,
            ASSISTANT_2_FINAL,
            _user(FOLLOWUP_QUESTION),
        ],
    },
]


# ── "Realistic" transitions: input-state pairs the workflow actually exercises ──
#
# These are the pairs the engine is *guaranteed* to see in a real multi-turn
# rollout — the input to turn k+1 is the input to turn k plus one or more
# messages, and both prefixes end on a non-assistant role. Property (iv)
# (sequence extension) is asserted on these and only these.

REALISTIC_TRANSITIONS_TOOL_VARIANT: list[tuple[str, str]] = [
    ("S0_system_plus_user", "S2_after_tool_response"),
]

REALISTIC_TRANSITIONS_USER_VARIANT: list[tuple[str, str]] = [
    ("S0_system_plus_user", "S2_after_tool_response"),
    ("S2_after_tool_response", "S4_after_followup_user"),
]


def get_chain(variant: str) -> list[ChainStep]:
    """Look up a chain by name. Useful for parametrized tests."""
    if variant == "ending_on_tool":
        return CHAIN_ENDING_ON_TOOL
    if variant == "ending_on_user":
        return CHAIN_ENDING_ON_USER
    raise ValueError(f"Unknown chain variant: {variant!r}")


def get_realistic_transitions(variant: str) -> list[tuple[str, str]]:
    if variant == "ending_on_tool":
        return REALISTIC_TRANSITIONS_TOOL_VARIANT
    if variant == "ending_on_user":
        return REALISTIC_TRANSITIONS_USER_VARIANT
    raise ValueError(f"Unknown chain variant: {variant!r}")


def step_by_label(chain: list[ChainStep], label: str) -> ChainStep:
    for step in chain:
        if step["label"] == label:
            return step
    raise KeyError(f"No step with label {label!r} in chain")


__all__ = [
    "ASSISTANT_1",
    "ASSISTANT_2_FINAL",
    "ASSISTANT_1_REASONING",
    "ASSISTANT_2_REASONING",
    "ASSISTANT_3_REASONING",
    "CHAIN_ENDING_ON_TOOL",
    "CHAIN_ENDING_ON_USER",
    "ChainStep",
    "FOLLOWUP_QUESTION",
    "REALISTIC_TRANSITIONS_TOOL_VARIANT",
    "REALISTIC_TRANSITIONS_USER_VARIANT",
    "SYSTEM_PROMPT",
    "TOOL_CALL_1",
    "TOOL_CALL_2",
    "TOOL_RESPONSE_1",
    "USER_QUESTION",
    "get_chain",
    "get_realistic_transitions",
    "step_by_label",
]
