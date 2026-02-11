#!/usr/bin/env python3
"""
Prerequisites:
1. Start the retrieval server:
   python examples/fully_async/deep_research/rag/rag_server.py --data_dir ./search_data/prebuilt_indices --port 9002

2. Set environment variables:
   export OPENAI_API_KEY="your-api-key"
   export RETRIEVAL_SERVER_URL="http://127.0.0.1:9002"

3. Run this script:
   python -m examples.fully_async.deep_research.search_agent
"""

import asyncio
import json
import time

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.fully_async.client import RolloutClient
from rllm.experimental.fully_async.protocol import Trajectory
from rllm.rewards.math_utils.utils import extract_boxed_answer
from rllm.rewards.reward_types import RewardConfig, RewardInput
from rllm.rewards.search_reward import RewardSearchFn

from .refine_agent import refine

MODEL = "Qwen/Qwen3-8B"
# MODEL = "gpt-oss-120b"
MAX_TURNS = 48
TOPP = 1.0
TEMP = 1.0

USE_REFINE = True
# RETRIEVAL_SERVER_URL = "http://<internal_ip>:9002"
# RETRIEVAL_SERVER_URL = "http://<internal_ip>:9002"
OVERLONG_FILTER = True  # overlong doesnt participate in loss calculation but still in advantage

TRAIN = True

if TRAIN:
    base_url = "http://localhost:4000"
    # base_url = "http://localhost:30001"
    api_key = ""
else:
    base_url = "http://localhost:30001"
    api_key = "token-abc123"

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


class SearchAgent:
    def __init__(self, use_tool=True, tool=None, model=MODEL, temperature=TEMP, top_p=TOPP, max_turns=MAX_TURNS, use_refine=USE_REFINE, client=None):
        self.llm = (
            RolloutClient(
                router_url=base_url,
                tokenizer=AutoTokenizer.from_pretrained(model),
            )
            if client is None
            else client
        )
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.tool = tool
        self.max_turns = max_turns
        self.use_tool = use_tool
        self.use_refine = use_refine

    async def generate(self, messages):
        kwargs = {
            # "model": self.model,
            # "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "extra_body": {
            "top_k": -1,  # Disable top_k sampling (vLLM defaults to 20)
            "repetition_penalty": 1.0,
            # "max_new_tokens": 8192,
            # },
        }
        tools = []
        if self.use_tool:
            tools = [self.tool.json]

        response, output = await self.llm.chat_completion(messages, sampling_params=kwargs, tools=tools)
        return response, len(output.all_response_ids()), output

    async def exec_tool_call(self, tool_calls: list[dict]):
        async def helper(tool_call):
            try:
                assert tool_call["function"]["name"] == self.tool.name
                args = tool_call["function"]["arguments"]
                args = json.loads(args)
                parse_error = 0
            except Exception as e:
                print("Error parsing tool call arguments:", e)
                args = {}
                parse_error = 1

            # Get query length for metrics
            query = args.get("query", "")
            query_length = len(query) if query else 0

            # tool.run returns "Error: ..." on failure
            tool_start_time = time.time()
            tool_result = await self.tool.run(**args)
            tool_wait_time = time.time() - tool_start_time
            tool_return_error = tool_result.startswith("Error:") * 1.0
            refine_time = 0
            refine_error = False
            if self.use_refine:
                before_len = len(tool_result)
                refine_start_time = time.time()
                tool_result = await refine(query=args.get("query"), result=tool_result)
                refine_time = time.time() - refine_start_time
                refine_error = len(tool_result) == before_len
                # print(f"Before {before_len}, After {len(tool_result)}")
            # Count tool result tokens using actual tokenizer
            tool_tokens = len(self.llm.tokenizer.encode(tool_result))
            metrics = {
                "tool_calls": 1,
                "parse_tool_args_error": parse_error,
                "tool_return_error": tool_return_error,
                "refine_error": refine_error,
                "tool_wait_time": tool_wait_time,
                "refine_time": refine_time,
                "query_length": query_length,
                "tool_tokens": tool_tokens,
            }
            # Format as message dict ready to append
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "content": tool_result,
            }
            return tool_message, metrics

        results = await asyncio.gather(*(helper(tc) for tc in tool_calls))
        combined_metrics = {"parse_tool_args_error": 0, "tool_return_error": 0, "refine_error": 0, "tool_calls": 0, "tool_wait_time": 0, "refine_time": 0, "query_length": 0, "tool_tokens": 0}
        tool_results = []
        for tr, m in results:
            tool_results.append(tr)
            for k in combined_metrics:
                combined_metrics[k] += m[k]
        return tool_results, combined_metrics

    async def run(self, question):
        trajectory = Trajectory(sequences=[], reward=0, metadata={})
        messages = [
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        metrics = []

        # Track executed search calls to detect duplicates
        executed_search_calls = set()
        duplicate_search_detected = False
        excessive_parallel_calls = False
        tool_error_detected = False
        refine_error_detected = False

        response = None
        num_turns = 0

        total_completion_tokens = 0
        total_generation_time = 0
        overlong = False

        try:
            for turn in range(self.max_turns):
                num_turns = turn + 1
                gen_start_time = time.time()
                response, completion_tokens, output = await self.generate(messages)
                total_generation_time += time.time() - gen_start_time
                messages.append(response)
                trajectory.append(output.to_sequence())
                total_completion_tokens += completion_tokens

                tool_calls = response.get("tool_calls", [])
                if tool_calls and turn < self.max_turns - 1:
                    # Check for excessive parallel tool calls (>= 5)
                    if len(tool_calls) >= 9:
                        excessive_parallel_calls = True
                        break

                    # Check for duplicate search calls before executing
                    for tool_call in tool_calls:
                        try:
                            call_key = (tool_call["function"]["name"], tool_call["function"]["arguments"])
                            if call_key in executed_search_calls:
                                duplicate_search_detected = True
                                break
                            executed_search_calls.add(call_key)
                        except (KeyError, TypeError):
                            pass

                    if duplicate_search_detected:
                        break

                    tool_messages, metric = await self.exec_tool_call(tool_calls)
                    messages.extend(tool_messages)
                    metrics.append(metric)

                    # Stop if any tool returned an error or refine error
                    if metric.get("tool_return_error", 0) > 0:
                        tool_error_detected = True
                        break
                    if metric.get("refine_error", 0) > 0:
                        refine_error_detected = True
                        break
                else:
                    break
        except Exception as e:
            import traceback

            overlong = True
            print("Error during agent run:", e)
            traceback.print_exc()

        # Parse final answer from response
        final_answer = None
        if response:
            content = response.get("content", "") or ""
            final_answer = extract_boxed_answer(content)

        # Aggregate metrics across all tool calls
        aggregated_metrics = {"num_turns": num_turns, "total_parse_tool_args_error": sum(m.get("parse_tool_args_error", 0) for m in metrics), "total_tool_return_error": sum(m.get("tool_return_error", 0) for m in metrics), "total_tool_calls": sum(m.get("tool_calls", 0) for m in metrics), "total_tool_wait_time": sum(m.get("tool_wait_time", 0) for m in metrics), "total_refine_time": sum(m.get("refine_time", 0) for m in metrics), "avg_refine_time": sum(m.get("refine_time", 0) for m in metrics) / max(sum(m.get("tool_calls", 0) for m in metrics), 1), "total_query_length": sum(m.get("query_length", 0) for m in metrics), "avg_query_length": sum(m.get("query_length", 0) for m in metrics) / max(sum(m.get("tool_calls", 0) for m in metrics), 1), "total_generation_time": total_generation_time, "total_completion_tokens": total_completion_tokens, "total_tool_tokens": sum(m.get("tool_tokens", 0) for m in metrics), "avg_completion_tokens_per_turn": total_completion_tokens / max(num_turns, 1), "avg_tool_tokens_per_call": sum(m.get("tool_tokens", 0) for m in metrics) / max(sum(m.get("tool_calls", 0) for m in metrics), 1), "duplicate_search_detected": duplicate_search_detected, "excessive_parallel_calls": excessive_parallel_calls, "tool_error_detected": tool_error_detected, "refine_error_detected": refine_error_detected, "overlong": overlong, "merged_step": len(trajectory.merge())}

        if OVERLONG_FILTER and overlong:
            for seq in trajectory.sequences:
                seq.response_masks = [0] * len(seq.response_masks)

        # Mask response when tool or refine returns error - don't train on this trajectory
        if tool_error_detected or refine_error_detected:
            for seq in trajectory.sequences:
                seq.response_masks = [0] * len(seq.response_masks)

        return {
            "trajectory": trajectory,
            "messages": messages,
            "final_answer": final_answer,
            "metrics": aggregated_metrics,
        }


def compute_reward(raw_response: str, ground_truth: str, timed_out: bool = False) -> dict:
    """Compute reward for the agent's answer using F1 score.

    Args:
        raw_response: The raw assistant response (reward fn will parse internally)
        ground_truth: The ground truth answer
        timed_out: Whether the agent timed out

    Returns:
        dict with is_correct, reward (F1 score), and metadata
    """
    reward_fn = RewardSearchFn(RewardConfig())

    if timed_out or not raw_response:
        return {
            "is_correct": False,
            "reward": 0.0,
            "metadata": {"reason": "timeout" if timed_out else "no_answer"},
        }

    # Pass raw response - reward fn will do internal parsing
    reward_input = RewardInput(task_info={"ground_truth": ground_truth}, action=raw_response)
    reward_output = reward_fn(reward_input)

    # Use F1 score directly as reward
    f1_score = reward_output.metadata.get("f1_score", 0.0) if reward_output.metadata else 0.0

    return {
        "is_correct": reward_output.is_correct,
        "reward": f1_score,  # Use F1 as reward
        "metadata": reward_output.metadata,
    }


def sanitize_metric(metric: dict) -> dict:
    """Sanitize metric dictionary to keep only float-convertible values."""
    sanitized = {}
    for k, v in metric.items():
        try:
            sanitized[k] = float(v)
        except (ValueError, TypeError):
            continue
    return sanitized


async def rollout(client, question: str, ground_truth: str, model=MODEL, max_turns=MAX_TURNS, use_refine=USE_REFINE, tool=None, **kwargs):
    assert tool is not None
    agent = SearchAgent(model=model, max_turns=max_turns, use_refine=use_refine, tool=tool, client=client)
    result = await agent.run(question)
    messages = result["messages"]
    trajectory = result["trajectory"]
    metrics = result["metrics"]

    # Check if duplicate search was detected - if so, set reward to 0
    if metrics.get("duplicate_search_detected", False):
        metrics["raw_reward"] = 0.0
        metrics["is_correct"] = False
        metrics = sanitize_metric(metrics)
        metrics["messages"] = messages
        metrics["trajectory"] = trajectory
        return 0.0, metrics

    # Check if excessive parallel tool calls detected (>= 5) - if so, set reward to 0
    if metrics.get("excessive_parallel_calls", False):
        metrics["raw_reward"] = 0.0
        metrics["is_correct"] = False
        metrics = sanitize_metric(metrics)
        metrics["messages"] = messages
        metrics["trajectory"] = trajectory
        return 0.0, metrics

    # Get raw response from last assistant message for reward computation
    raw_response = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            raw_response = msg.get("content", "") or ""
            if raw_response:
                break

    reward_info = compute_reward(raw_response, ground_truth)

    # Only keep metadata values that can be converted to float (numbers, bools, numeric strings)
    metrics.update(reward_info["metadata"])
    metrics["raw_reward"] = reward_info["reward"]
    metrics["is_correct"] = reward_info["is_correct"]
    metrics = sanitize_metric(metrics)
    metrics["messages"] = messages
    metrics["trajectory"] = trajectory

    return metrics["raw_reward"], metrics


if __name__ == "__main__":
    dataset = DatasetRegistry.load_dataset("browsecomp-plus", split="test")

    async def run_sample(sample):
        question = sample["question"]
        agent = SearchAgent(model=MODEL, max_turns=MAX_TURNS, use_refine=USE_REFINE)
        result = await agent.run(question)

        if len(result["messages"]) > 3:
            print(result)

    async def run(samples):
        return await asyncio.gather(*[run_sample(sample) for sample in samples])

    asyncio.run(run(dataset[:20]))
