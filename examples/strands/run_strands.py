"""Strands RAG Agent with training interface."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime

from dotenv import find_dotenv, load_dotenv

try:
    from .retrieve_tool import retrieve
    from .strands_workflow import StrandsWorkflow
except ImportError:
    from retrieve_tool import retrieve
    from strands_workflow import StrandsWorkflow

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent
from rllm.rewards.reward_fn import RewardInput
from rllm.rewards.search_reward import RewardConfig, RewardSearchFn

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("strands.telemetry").setLevel(logging.CRITICAL)


# Global singleton for engine
_ROLLOUT_ENGINE = None


def get_rollout_engine():
    """Get or create the rollout engine (singleton)."""
    global _ROLLOUT_ENGINE
    if _ROLLOUT_ENGINE is None:
        _ROLLOUT_ENGINE = OpenAIEngine(
            model=os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B"),
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:4000/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            sampling_params={"temperature": 0.7, "max_tokens": 1024},
        )
    return _ROLLOUT_ENGINE


TRAINING_SYSTEM_PROMPT = """You are a helpful AI assistant that can search for information to answer questions accurately.

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


async def run_search_agent(question: str, ground_truth: str) -> dict:
    """Training interface for AgentTrainer."""
    rollout_engine = get_rollout_engine()
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")

    model = RLLMModel(rollout_engine=rollout_engine, model_id=model_name)
    agent = StrandsAgent(
        model=model,
        tools=[retrieve],
        system_prompt=TRAINING_SYSTEM_PROMPT,
    )

    try:
        result = agent(question)
        content = str(result)
        match = re.search(r"\\boxed\{([^}]+)\}", content)
        final_answer = match.group(1) if match else content.strip()
    except Exception as e:
        print(f"Agent execution failed: {e}")
        return {"reward": 0.0, "answer": ""}

    reward_fn = RewardSearchFn(RewardConfig())
    reward_output = reward_fn(
        RewardInput(
            task_info={"ground_truth": ground_truth},
            action=final_answer,
        )
    )

    return {"reward": reward_output.reward, "answer": final_answer}


def save_episode_to_json(episode, output_dir="./strands_outputs"):
    """Save episode trajectory to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    trajectory = episode.trajectories[0] if episode.trajectories else None
    if not trajectory:
        print("Error: No trajectory found")
        return None

    trajectory_data = {
        "task": trajectory.task,
        "reward": trajectory.reward,
        "steps": [],
        "tool_calls_summary": [],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_steps": len(trajectory.steps),
            "total_tool_calls": 0,
            "agent_type": "StrandsAgent",
            "episode_id": episode.id,
        },
    }

    tool_call_count = 0
    for i, step in enumerate(trajectory.steps):
        step_data = {
            "step_index": i,
            "observation": step.observation,
            "model_response": step.model_response,
            "action": step.action,
            "reward": step.reward,
            "done": step.done,
            "chat_completions": step.chat_completions,
        }

        if step.action and isinstance(step.action, dict):
            if step.action.get("type") == "tool_calls":
                step_data["step_type"] = "tool_calls"
                step_data["tool_calls"] = step.action.get("tool_calls", [])
                tool_call_count += len(step.action.get("tool_calls", []))
            elif step.action.get("type") == "tool_call":
                step_data["step_type"] = "tool_call"
                step_data["tool_name"] = step.action.get("tool_name")
                step_data["tool_args"] = step.action.get("tool_args")
                step_data["tool_result"] = step.action.get("tool_result")
                tool_call_count += 1
            else:
                step_data["step_type"] = "conversation"
        else:
            step_data["step_type"] = "conversation"

        trajectory_data["steps"].append(step_data)

    trajectory_data["metadata"]["total_tool_calls"] = tool_call_count

    tool_calls_summary = []
    for step in trajectory.steps:
        if step.action and isinstance(step.action, dict):
            if step.action.get("type") == "tool_calls":
                for tool_call in step.action.get("tool_calls", []):
                    tool_calls_summary.append({
                        "tool_name": tool_call.get("name"),
                        "tool_args": tool_call.get("input"),
                        "tool_id": tool_call.get("id"),
                    })
            elif step.action.get("type") == "tool_call":
                tool_calls_summary.append({
                    "tool_name": step.action.get("tool_name"),
                    "tool_args": step.action.get("tool_args"),
                    "tool_result": step.action.get("tool_result"),
                })
    trajectory_data["tool_calls_summary"] = tool_calls_summary

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"strands_trajectory_{timestamp}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

    print(f"Trajectory saved to: {filepath}")
    print(f"Summary: {len(trajectory.steps)} steps, {tool_call_count} tool calls, reward: {trajectory.reward}")

    return filepath


async def run_strands_workflow(rollout_engine):
    """Run StrandsWorkflow with AgentWorkflowEngine."""
    model = RLLMModel(rollout_engine=rollout_engine, model_id="Qwen/Qwen3-0.6B")

    system_prompt = TRAINING_SYSTEM_PROMPT

    workflow_engine = AgentWorkflowEngine(
        workflow_cls=StrandsWorkflow,
        workflow_args={
            "agent_cls": StrandsAgent,
            "agent_args": {"model": model, "tools": [retrieve], "system_prompt": system_prompt},
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=8,
    )

    await workflow_engine.initialize_pool()

    task = os.getenv("STRANDS_TASK", "Who founded Stripe and when was the company started?")
    task_dict = {"task": task}
    task_id = "strands_task_001"

    print(f"Task: {task}")
    print(f"Available tools: ['retrieve']")
    print("\n" + "=" * 80)
    print("Starting workflow execution...")
    print("=" * 80)

    episodes = await workflow_engine.execute_tasks([task_dict], [task_id])

    if not episodes:
        print("Error: No episodes returned")
        return

    episode = episodes[0]

    print("\n" + "=" * 80)
    print("Workflow execution completed!")
    print("=" * 80)

    save_episode_to_json(episode)

    print(f"\nEpisode {episode.id} completed")
    for i, trajectory in enumerate(episode.trajectories):
        print(f"Trajectory {i}: {len(trajectory.steps)} steps, reward: {trajectory.reward}")


async def main():
    load_dotenv(find_dotenv())

    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if together_api_key:
        base_url = "https://api.together.xyz/v1"
        api_key = together_api_key
        model_id = os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    elif openai_api_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = openai_api_key
        model_id = os.getenv("MODEL_NAME", "gpt-4o")
    else:
        raise ValueError("API key required (TOGETHER_AI_API_KEY or OPENAI_API_KEY)")

    rollout_engine = OpenAIEngine(
        model=model_id,
        tokenizer=None,
        base_url=base_url,
        api_key=api_key,
        sampling_params={"temperature": 0.7, "top_p": 0.95, "max_tokens": 512},
    )

    await run_strands_workflow(rollout_engine)


if __name__ == "__main__":
    asyncio.run(main())
