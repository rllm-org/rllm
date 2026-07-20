"""Verify native_react message and token parity against a live Qwen3.5 deployment.

The chat requests verify native response fields and measure whether the public
chat formatter consumes resent ``reasoning_content``. The decisive parity check
uses Fireworks' real token-in endpoint—the path used by VMVM-v2 and rLLM
training—to send exact locally rendered IDs for two turns and compare server
prompt counts plus bridge/full-render identity.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests
from fireworks.training.sdk import DeploymentSampler
from renderers import create_renderer
from transformers import AutoTokenizer

from rllm.harnesses.native_react import (
    NATIVE_TOOL_SCHEMAS,
    initial_messages,
    preserve_assistant_message,
    tool_observation,
)

DEFAULT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_INFERENCE_URL = "https://api.fireworks.ai"
DEFAULT_MODEL = os.environ.get("RLLM_VERIFY_MODEL")
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-35B-A3B"


def _api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key
    config_path = Path.home() / ".rllm" / "config.json"
    if config_path.exists():
        key = json.loads(config_path.read_text()).get("api_keys", {}).get("fireworks")
    if not key:
        raise RuntimeError("Set FIREWORKS_API_KEY or configure api_keys.fireworks in ~/.rllm/config.json")
    return key


def _post(url: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        json=payload,
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Fireworks returned HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def _payload(model: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "messages": messages,
        "tools": NATIVE_TOOL_SCHEMAS,
    }


def _message(response: dict[str, Any]) -> dict[str, Any]:
    return preserve_assistant_message(response["choices"][0]["message"])


def _prompt_tokens(response: dict[str, Any]) -> int:
    return int(response["usage"]["prompt_tokens"])


def _token_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _parsed_assistant_message(parsed: Any) -> dict[str, Any]:
    """Convert a renderer response to the same structured assistant shape."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": parsed.content,
    }
    if parsed.reasoning_content is not None:
        message["reasoning_content"] = parsed.reasoning_content
    if parsed.tool_calls:
        calls = []
        for index, tool_call in enumerate(parsed.tool_calls):
            function = tool_call.get("function") or tool_call
            arguments = function.get("arguments", {})
            calls.append(
                {
                    "id": tool_call.get("id") or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": (arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)),
                    },
                }
            )
        message["tool_calls"] = calls
    return message


def _tool_names(parsed: Any) -> list[str]:
    names = []
    for tool_call in parsed.tool_calls or []:
        function = tool_call.get("function") or tool_call
        names.append(str(function.get("name", "")))
    return names


async def _sample_token_prompt(
    sampler: DeploymentSampler,
    prompt_ids: list[int],
    *,
    max_tokens: int,
) -> tuple[dict[str, Any], Any, list[int]]:
    response, metrics = await sampler.async_completions_stream(
        prompt=prompt_ids,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1.0,
        top_k=40,
        raw_output=True,
        logprobs=True,
        http_timeout=180,
    )
    choice = response["choices"][0]
    completion_ids = list((choice.get("raw_output") or {}).get("completion_token_ids") or [])
    if not completion_ids:
        raise AssertionError("token-in deployment response omitted completion_token_ids")
    if metrics.prompt_tokens != len(prompt_ids):
        raise AssertionError(f"deployment received {metrics.prompt_tokens} prompt tokens; sent {len(prompt_ids)}")
    return response, metrics, completion_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--inference-url", default=DEFAULT_INFERENCE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL, required=DEFAULT_MODEL is None)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    key = _api_key()
    messages = initial_messages(
        "Call the bash tool with the command `pwd`. After seeing its output, call the submit tool.",
        "/app\nfile.txt",
    )
    turn_one = _post(args.url, key, _payload(args.model, messages, 512))
    assistant = _message(turn_one)
    if not assistant.get("reasoning_content"):
        raise AssertionError("turn one did not return reasoning_content")
    if not assistant.get("tool_calls"):
        raise AssertionError("turn one did not return a structured tool call")

    observation = tool_observation("Current terminal state:\n/app")
    turn_two_messages = messages + [assistant, observation]
    turn_two = _post(args.url, key, _payload(args.model, turn_two_messages, 512))
    second_assistant = _message(turn_two)
    if not second_assistant.get("tool_calls"):
        raise AssertionError("turn two did not return a structured tool call")
    if turn_two_messages[-2] != assistant:
        raise AssertionError("turn-one assistant message changed before turn two")

    no_reasoning_assistant = dict(assistant)
    no_reasoning_assistant.pop("reasoning_content", None)
    no_reasoning_assistant.pop("reasoning", None)
    control_messages = messages + [no_reasoning_assistant, observation]
    control = _post(args.url, key, _payload(args.model, control_messages, 1))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    renderer = create_renderer(tokenizer, renderer="qwen3.5")
    live_ids = renderer.render_ids(turn_two_messages, tools=NATIVE_TOOL_SCHEMAS, add_generation_prompt=True)
    control_ids = renderer.render_ids(control_messages, tools=NATIVE_TOOL_SCHEMAS, add_generation_prompt=True)
    live_prompt_tokens = _prompt_tokens(turn_two)
    control_prompt_tokens = _prompt_tokens(control)

    # Fireworks' public chat endpoint applies its own tool-schema normalization
    # and currently ignores resent reasoning_content.  Keep this control in the
    # report so the distinction is explicit; the token-in checks below are the
    # training-path preservation guarantee.
    live_delta = live_prompt_tokens - control_prompt_tokens
    local_delta = len(live_ids) - len(control_ids)

    turn_one_prompt = renderer.render_ids(messages, tools=NATIVE_TOOL_SCHEMAS, add_generation_prompt=True)
    closed_turn_one = renderer.render_ids(messages + [assistant], tools=NATIVE_TOOL_SCHEMAS)
    completion_ids = closed_turn_one[len(turn_one_prompt) :]
    bridged = renderer.bridge_to_next_turn(
        turn_one_prompt,
        completion_ids,
        [observation],
        tools=NATIVE_TOOL_SCHEMAS,
    )
    if bridged is None or bridged.token_ids != live_ids:
        raise AssertionError("cumulative bridge tokens differ from the VMVM-v2 full render")

    # PRIME VMVM-v2 training and rLLM training both render client-side and send
    # integer token IDs to /inference/v1/completions.  Exercise that exact path
    # against the real deployment for two turns.  Server metrics independently
    # confirm the number of IDs received on each request.
    sampler = DeploymentSampler(
        args.inference_url,
        args.model,
        key,
        tokenizer=tokenizer,
    )
    loop = asyncio.new_event_loop()
    try:
        token_turn_one, token_metrics_one, token_completion_one = loop.run_until_complete(_sample_token_prompt(sampler, turn_one_prompt, max_tokens=512))
        parsed_one = renderer.parse_response(token_completion_one)
        token_tool_names_one = _tool_names(parsed_one)
        if not parsed_one.reasoning_content:
            raise AssertionError("token-in turn one did not produce interleaved reasoning")
        if "bash" not in token_tool_names_one:
            raise AssertionError(f"token-in turn one did not call bash: {token_tool_names_one}")

        token_assistant_one = _parsed_assistant_message(parsed_one)
        token_turn_two_messages = messages + [token_assistant_one, observation]
        token_full_turn_two = renderer.render_ids(
            token_turn_two_messages,
            tools=NATIVE_TOOL_SCHEMAS,
            add_generation_prompt=True,
        )
        token_bridge = renderer.bridge_to_next_turn(
            turn_one_prompt,
            token_completion_one,
            [observation],
            tools=NATIVE_TOOL_SCHEMAS,
        )
        if token_bridge is None or token_bridge.token_ids != token_full_turn_two:
            raise AssertionError("real completion bridge differs from the VMVM-v2 full turn-two render")

        token_turn_two, token_metrics_two, token_completion_two = loop.run_until_complete(_sample_token_prompt(sampler, token_bridge.token_ids, max_tokens=256))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    parsed_two = renderer.parse_response(token_completion_two)
    token_tool_names_two = _tool_names(parsed_two)
    if "submit" not in token_tool_names_two:
        raise AssertionError(f"token-in turn two did not call submit: {token_tool_names_two}")

    reasoning = str(assistant["reasoning_content"])
    token_reasoning = str(parsed_one.reasoning_content)
    report = {
        "ok": True,
        "model": args.model,
        "turn_one_message_fields": sorted(assistant),
        "turn_two_message_fields": sorted(second_assistant),
        "turn_one_finish_reason": turn_one["choices"][0].get("finish_reason"),
        "turn_two_finish_reason": turn_two["choices"][0].get("finish_reason"),
        "turn_one_reasoning_chars": len(reasoning),
        "turn_one_reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "turn_two_reasoning_chars": len(str(second_assistant.get("reasoning_content") or "")),
        "turn_two_prompt_tokens": live_prompt_tokens,
        "turn_two_without_reasoning_prompt_tokens": control_prompt_tokens,
        "chat_resent_reasoning_token_delta": live_delta,
        "vmvm_reasoning_token_delta": local_delta,
        "chat_endpoint_preserves_resent_reasoning": live_delta == local_delta and live_delta > 0,
        "chat_prompt_count_offset_from_vmvm_render": len(live_ids) - live_prompt_tokens,
        "chat_bridge_equals_vmvm_full_render": True,
        "token_in_turn_one_prompt_tokens_sent": len(turn_one_prompt),
        "token_in_turn_one_prompt_tokens_received": token_metrics_one.prompt_tokens,
        "token_in_turn_one_prompt_sha256": _token_sha256(turn_one_prompt),
        "token_in_turn_one_completion_tokens": len(token_completion_one),
        "token_in_turn_one_finish_reason": token_turn_one["choices"][0].get("finish_reason"),
        "token_in_turn_one_reasoning_chars": len(token_reasoning),
        "token_in_turn_one_reasoning_sha256": hashlib.sha256(token_reasoning.encode()).hexdigest(),
        "token_in_turn_one_tools": token_tool_names_one,
        "token_in_turn_two_prompt_tokens_sent": len(token_bridge.token_ids),
        "token_in_turn_two_prompt_tokens_received": token_metrics_two.prompt_tokens,
        "token_in_turn_two_prompt_sha256": _token_sha256(token_bridge.token_ids),
        "token_in_turn_two_completion_tokens": len(token_completion_two),
        "token_in_turn_two_finish_reason": token_turn_two["choices"][0].get("finish_reason"),
        "token_in_turn_two_tools": token_tool_names_two,
        "real_completion_bridge_equals_vmvm_full_render": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
