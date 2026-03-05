#!/usr/bin/env python3
# Suppress "Failed to detach context" log spam from Strands SDK async generators.
# These are harmless errors when breaking from stream_async() early (GeneratorExit).
# We suppress the LOGGER only — the real OpenTelemetry tracer stays active for rLLM token capture.
# (Do NOT use NoOpTracerProvider — that disables the tracer entirely and breaks token capture.)
import logging
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

"""
Strands-based search agent for HotPotQA using local retrieval server.

This is a Strands implementation similar to examples/sdk/langgraph/search_agent_langgraph.py
but using Strands Agent framework instead of LangGraph.

Prerequisites:
1. Start the RAG server:
   cd examples/strands/rag && bash launch_rag.sh ./search_data/prebuilt_indices 9002

2. Set environment variables:
   export RETRIEVAL_SERVER_URL="http://127.0.0.1:9002"

3. Run this script:
   python examples/strands/search_agent_strands.py
"""

import asyncio
import os
import re
import time

# Strands imports
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.handlers.callback_handler import null_callback_handler
from strands.event_loop.event_loop import MaxTokensReachedException

# rLLM SDK imports
from rllm.rewards.reward_fn import RewardInput
from rllm.rewards.search_reward import RewardConfig, RewardSearchFn
from rllm.sdk import get_chat_client_async
from rllm.sdk.session.base import _ensure_tracer_initialized

# Import the local search tool
from retrieve_tool import local_search

MODEL = os.environ.get("STRANDS_MODEL", "Qwen/Qwen3-4B")
# NOTE: This is the EFFECTIVE step limit. rllm.agent.max_steps=10 in the shell script
# is a dead config — AgentSdkEngine never forwards it to run_agent(). Both Strands and
# LangGraph hardcode MAX_TURNS=5 here. 1 turn = 1 LLM call (not 1 tool call).
MAX_TURNS = 5
MAX_RESPONSE_TOKENS = 2048

TRAIN = os.environ.get("TRAIN", "0") == "1"

if TRAIN:
    # Initialize tracer FIRST before creating client (critical for session context)
    _ensure_tracer_initialized("strands_search_agent")
    base_url = "http://localhost:4000/v1"  # rLLM LiteLLM proxy for token capture
    api_key = ""
    use_proxy = True
else:
    # Check for external API mode (Together AI or OpenAI)
    if os.environ.get("TOGETHER_API_KEY"):
        base_url = os.environ.get("VLLM_URL", "https://api.together.xyz/v1")
        api_key = os.environ.get("TOGETHER_API_KEY")
        use_proxy = False
    elif os.environ.get("OPENAI_API_KEY"):
        base_url = os.environ.get("VLLM_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
        MODEL = os.environ.get("MODEL", "gpt-4o-mini")  # Override model for OpenAI
        use_proxy = False
    else:
        # Default: local vLLM server
        base_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
        api_key = "token-abc123"
        use_proxy = False

SEARCH_SYSTEM_PROMPT = """You are a helpful AI assistant that can search for information to answer questions accurately.

When answering questions:
1. Use the available search tools to find relevant and reliable information
2. Synthesize information from multiple sources when needed
3. Provide accurate and comprehensive answers based on your search results
4. Always put your final answer in \\boxed{} format

For example:
- If the answer is "American", write: \\boxed{American}
- If the answer is "yes", write: \\boxed{yes}
- If the answer is a year like "1985", write: \\boxed{1985}

Remember to search thoroughly and provide your final answer clearly within the \\boxed{} format."""


class NonStreamingOpenAIModel(OpenAIModel):
    """OpenAIModel subclass that forces non-streaming API calls.

    Strands SDK hardcodes stream=True in format_request(), but LiteLLM proxy's
    async_post_call_success_hook only fires for non-streaming requests. Streaming
    requests return StreamingResponse immediately, skipping the hook entirely,
    so zero traces are written to SQLite and training crashes with empty sequences.

    This subclass:
    1. Overrides format_request() to set stream=False
    2. Overrides stream() to make a non-streaming call, then yield StreamEvent
       dicts in the format the Strands event loop expects
    """

    def format_request(self, messages, tool_specs=None, system_prompt=None,
                       tool_choice=None, **kwargs):
        request = super().format_request(messages, tool_specs, system_prompt,
                                         tool_choice, **kwargs)
        request["stream"] = False
        request.pop("stream_options", None)
        return request

    async def stream(self, messages, tool_specs=None, system_prompt=None,
                     *, tool_choice=None, **kwargs):
        """Non-streaming version that converts ChatCompletion to StreamEvents."""
        import openai
        from strands.models.openai import ContextWindowOverflowException, ModelThrottledException

        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)

        async with self._get_client() as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except openai.RateLimitError as e:
                raise ModelThrottledException(str(e)) from e

            yield {"messageStart": {"role": "assistant"}}

            choice = response.choices[0]
            message = choice.message

            # Log tool call detection for debugging (Track 2: root cause investigation)
            # If vLLM's Hermes parser fails (929 JSONDecodeErrors in logs), tool_calls will be
            # None/empty and finish_reason will be "stop" instead of "tool_calls"
            if os.environ.get("RLLM_SDK_DIAGNOSTICS") == "1":
                has_tools = bool(message.tool_calls)
                n_tools = len(message.tool_calls) if message.tool_calls else 0
                print(f"[VLLM_RESPONSE] finish_reason={choice.finish_reason}, "
                      f"has_tool_calls={has_tools}, n_tool_calls={n_tools}, "
                      f"content_len={len(message.content) if message.content else 0}")

            # Yield text content
            if message.content:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": message.content}}}
                yield {"contentBlockStop": {}}

            # Yield tool calls
            if message.tool_calls:
                for tc in message.tool_calls:
                    yield {"contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": tc.id, "name": tc.function.name}}
                    }}
                    yield {"contentBlockDelta": {
                        "delta": {"toolUse": {"input": tc.function.arguments}}
                    }}
                    yield {"contentBlockStop": {}}

            # Map OpenAI finish_reason to Strands stopReason
            stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
            stop_reason = stop_map.get(choice.finish_reason, "end_turn")
            yield {"messageStop": {"stopReason": stop_reason}}

            # Usage metadata
            if response.usage:
                yield {"metadata": {
                    "usage": {
                        "inputTokens": response.usage.prompt_tokens,
                        "outputTokens": response.usage.completion_tokens,
                        "totalTokens": response.usage.total_tokens,
                    },
                    "metrics": {"latencyMs": 0},
                }}


# Initialize the Strands model with RLLM SDK
# Use async client for token capture during training
_traced_client = get_chat_client_async(
    api_key=api_key,
    base_url=base_url,
    use_proxy=use_proxy,
)

# Pass traced client to NonStreamingOpenAIModel - forces stream=False so LiteLLM
# proxy's async_post_call_success_hook fires and traces are written to SQLite
_model = NonStreamingOpenAIModel(
    model_id=MODEL,
    client=_traced_client,
    params={"temperature": float(os.environ.get("STRANDS_TEMPERATURE", "0.7")), "max_tokens": MAX_RESPONSE_TOKENS},
)


# Pre-build shared agent config at module level to avoid repeated setup per rollout.
# NOTE: We cannot share a single Agent instance across concurrent rollouts because
# Strands Agent accumulates conversation state (messages) internally. Each rollout
# needs its own Agent. But the expensive parts (model, tool list) are already shared.
_AGENT_TOOLS = [local_search]
_AGENT_KWARGS = dict(
    model=_model,
    tools=_AGENT_TOOLS,
    system_prompt=SEARCH_SYSTEM_PROMPT,
    callback_handler=null_callback_handler,
)


async def run_search_agent(question: str, ground_truth: str = None, max_turns: int = MAX_TURNS) -> dict:
    """
    Run the Strands search agent on a question.

    Args:
        question: The question to answer
        ground_truth: Optional ground truth for reward calculation
        max_turns: Maximum agent turns (default: 5, matches LangGraph)

    Returns:
        dict with keys: question, final_answer, ground_truth, reward, is_correct, messages, num_turns
    """
    # Create agent per rollout (required: Agent holds conversation state).
    # Model + tools are module-level singletons — only the lightweight Agent wrapper is new.
    agent = Agent(**_AGENT_KWARGS)

    messages = []
    final_answer = None
    num_tool_turns = 0
    num_model_turns = 0
    timed_out = False
    content = ""

    # Timing instrumentation
    episode_start = time.perf_counter()
    tool_times = []
    _tool_start = None

    # NOTE: No max_iterations counter here! Unlike LangGraph (which streams full node steps),
    # Strands streams token-by-token. A limit of 15 events would kill the stream after ~15 tokens.
    # Budget enforcement is via num_tool_turns > max_turns (see is_tool_use block below).
    # model_turns is always 0 — Strands SDK consumes messageStart/messageStop internally.

    # Use stream_async for async iteration (Strands equivalent of LangGraph's astream)
    stream = agent.stream_async(question)
    try:
        async for event in stream:
            messages.append(event)

            # Count tool calls (Strands uses current_tool_use with name property)
            is_tool_use = (isinstance(event, dict) and
                          "current_tool_use" in event and
                          event.get("current_tool_use", {}).get("name"))
            if is_tool_use:
                # Record elapsed time for previous tool call (if any)
                if _tool_start is not None:
                    tool_times.append(time.perf_counter() - _tool_start)
                num_tool_turns += 1
                _tool_start = time.perf_counter()
                tool_name = event.get("current_tool_use", {}).get("name", "unknown")
                # Log first few tool calls for debugging tool usage
                if num_tool_turns <= 3:
                    print(f"[TOOL_USE #{num_tool_turns}] tool={tool_name}")
                # Enforce budget on tool turns (model_turns is always 0 — Strands SDK bug).
                # Allow max_turns tool calls (5), break on the (max_turns+1)th.
                # Matches LangGraph: 5 LLM calls allowed, ~4 with tools + 1 answer.
                if num_tool_turns > max_turns:
                    timed_out = True
                    break

            # Count model turns via messageStop events (one per model response)
            # NOTE: num_model_turns is always 0 — Strands SDK consumes messageStop internally.
            # Budget enforcement is in the is_tool_use block above (via num_tool_turns).
            is_message_stop = isinstance(event, dict) and "messageStop" in event
            if is_message_stop:
                num_model_turns += 1

            # Extract content from event (Strands uses contentBlockDelta for streaming text)
            # Event structure: {"contentBlockDelta": {"delta": {"text": "actual text here"}}}
            if isinstance(event, dict) and "contentBlockDelta" in event:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                event_content = delta.get("text", "")
            elif isinstance(event, dict) and "data" in event:
                event_content = str(event["data"])
            else:
                event_content = ""

            content += event_content  # Accumulate (not overwrite) for regex matching

            # Extract final answer if present (run on accumulated content, not single chunk)
            # Strands streams token-by-token, so \boxed{} may be split across events
            match = re.search(r"\\boxed\{([^}]+)\}", content)
            if match:
                final_answer = match.group(1)

    except MaxTokensReachedException:
        # Model hit token limit — treat as timeout, don't crash.
        # Content accumulated so far is kept for reward evaluation.
        pass
    except Exception as e:
        print(f"Agent error: {e}")
        content = ""
        final_answer = None
    finally:
        # Explicitly close the async generator to prevent resource leaks
        # (GeneratorExit from breaking early can cause "Failed to detach context" warnings)
        try:
            await stream.aclose()
        except Exception:
            pass
        # Eagerly release Strands Agent state before GC collects it.
        # During a rollout the agent accumulates messages/traces internally;
        # with 512 concurrent rollouts that's 512 live agent objects worth of
        # state. LangGraph doesn't have this problem — its graph is a
        # module-level singleton and each call passes fresh state in/out,
        # so nothing accumulates on the graph object itself.
        try:
            agent.messages.clear()
            agent.cleanup()
        except Exception:
            pass

    # Record final tool time if a tool was still running when we broke out
    if _tool_start is not None:
        tool_times.append(time.perf_counter() - _tool_start)

    # Per-sample trajectory log — one line per sample, grep with [TRAJ]
    episode_time = time.perf_counter() - episode_start
    tool_total = sum(tool_times)
    avg_tool_ms = (tool_total / len(tool_times) * 1000) if tool_times else 0
    print(
        f"[TRAJ] events={len(messages)} model_turns={num_model_turns} "
        f"tool_turns={num_tool_turns} answer={'yes' if final_answer else 'no'} "
        f"timeout={'yes' if timed_out else 'no'} content_len={len(content)} "
        f"episode={episode_time:.1f}s avg_tool={avg_tool_ms:.0f}ms "
        f"tool_total={tool_total:.1f}s"
    )

    result = {
        "question": question,
        "final_answer": final_answer,
        "ground_truth": ground_truth,
        "full_response": content,
        "messages": messages,
        "num_turns": num_model_turns,  # backward compat: report model turns as "num_turns"
        "num_tool_turns": num_tool_turns,
        "num_model_turns": num_model_turns,
        "timed_out": timed_out,
    }

    # Evaluate if ground truth is provided
    if ground_truth:
        reward_fn = RewardSearchFn(RewardConfig())

        # If timed out or no answer, score is 0
        if timed_out or not final_answer:
            result["is_correct"] = False
            result["reward"] = 0.0
            result["evaluation_metadata"] = {"reason": "timeout" if timed_out else "no_answer"}
        else:
            # Normal evaluation
            reward_input = RewardInput(
                task_info={"ground_truth": ground_truth},
                action=final_answer
            )
            reward_output = reward_fn(reward_input)

            result["is_correct"] = reward_output.is_correct
            result["reward"] = reward_output.reward
            result["evaluation_metadata"] = reward_output.metadata
    else:
        result["reward"] = 0.0
        result["is_correct"] = False

    return result


async def main():
    """Main async entry point."""
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:9002"

    from rllm.data import DatasetRegistry

    test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    results = []
    for task in test_dataset.get_data()[:5]:  # Run on first 5 for demo
        result = await run_search_agent(question=task["question"], ground_truth=task.get("ground_truth"))
        results.append(result)

        print(f"\nQuestion: {result['question'][:100]}...")
        print(f"Answer: {result['final_answer']}")
        print(f"Correct: {result.get('is_correct', 'N/A')}")

    # Summary statistics
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {correct_count}/{len(results)} correct")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
