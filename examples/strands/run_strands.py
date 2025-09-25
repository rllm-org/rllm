import asyncio
import json
import logging
import os
from datetime import datetime

from dotenv import find_dotenv, load_dotenv
from gsearch_tool_wrapped import google_search
from strands_tools import calculator, file_read, http_request, python_repl
from strands_tools.browser import LocalChromiumBrowser
from strands_workflow import StrandsWorkflow

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent

# Disable OpenTelemetry SDK
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# Disable OpenTelemetry error logs
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("strands.telemetry").setLevel(logging.CRITICAL)


def save_episode_to_json(episode, output_dir="./strands_outputs"):
    """Save the episode to a JSON file with timestamp."""
    os.makedirs(output_dir, exist_ok=True)

    # Get trajectory data from episode
    trajectory = episode.trajectories[0][1] if episode.trajectories else None
    if not trajectory:
        print("❌ No trajectory found in episode")
        return None

    # Convert trajectory to dict format
    trajectory_data = {"task": trajectory.task, "reward": trajectory.reward, "steps": [], "tool_calls_summary": [], "metadata": {"timestamp": datetime.now().isoformat(), "total_steps": len(trajectory.steps), "total_tool_calls": 0, "agent_type": "StrandsAgent", "episode_id": episode.id}}

    # Convert each step to dict and count tool calls
    tool_call_count = 0
    for i, step in enumerate(trajectory.steps):
        step_data = {"step_index": i, "observation": step.observation, "model_response": step.model_response, "action": step.action, "reward": step.reward, "done": step.done, "chat_completions": step.chat_completions}

        # Add tool call specific information
        if step.action and isinstance(step.action, dict):
            # 🔄 Support new tool_calls format (hybrid approach)
            if step.action.get("type") == "tool_calls":
                step_data["step_type"] = "tool_calls"
                step_data["tool_calls"] = step.action.get("tool_calls", [])
                tool_call_count += len(step.action.get("tool_calls", []))
            # 🔙 Support legacy tool_call format (backup compatibility)
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

    # Update tool call count
    trajectory_data["metadata"]["total_tool_calls"] = tool_call_count

    # Generate tool calls summary from both formats
    tool_calls_summary = []
    for step in trajectory.steps:
        if step.action and isinstance(step.action, dict):
            # 🔄 Support new tool_calls format
            if step.action.get("type") == "tool_calls":
                for tool_call in step.action.get("tool_calls", []):
                    tool_calls_summary.append({"tool_name": tool_call.get("name"), "tool_args": tool_call.get("input"), "tool_id": tool_call.get("id")})
            # Support legacy tool_call format (backward compatibility)
            elif step.action.get("type") == "tool_call":
                tool_calls_summary.append({"tool_name": step.action.get("tool_name"), "tool_args": step.action.get("tool_args"), "tool_result": step.action.get("tool_result")})
    trajectory_data["tool_calls_summary"] = tool_calls_summary

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strands_trajectory_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Save to JSON file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

    print(f"📁 Trajectory saved to: {filepath}")

    # Print concise summary
    print(f"📊 Summary: {len(trajectory.steps)} steps, {tool_call_count} tool calls, reward: {trajectory.reward}")

    return filepath


async def run_strands_workflow(rollout_engine):
    """Example using StrandsWorkflow with AgentWorkflowEngine."""

    # Create RLLMModel
    model = RLLMModel(rollout_engine=rollout_engine, model_id="Qwen/Qwen3-0.6B")

    # Initialize browser tool (graceful fallback if runtime not installed)
    try:
        browser = LocalChromiumBrowser()
        browser_tool = browser.browser
    except Exception as e:
        logging.warning(f"Browser disabled: {e}")
        browser = None
        browser_tool = None

    # Prepare tool set with all available tools
    tools = [
        calculator,  # strands_tools calculator
        http_request,  # Native strands format with TOOL_SPEC
        file_read,  # Native strands format with TOOL_SPEC
        python_repl,  # Native strands format with TOOL_SPEC
        google_search,  # Custom tool
    ]
    # Disable browser tool for now
    if browser_tool:
        tools.append(browser_tool)

    # System prompt with all available tools
    system_prompt = """You are an expert agent solving web-based tasks from the Gaia dataset. 
        These tasks often require searching for current information, analyzing data, 
        and providing accurate answers. Use the available tools when needed:
        - calculator: for mathematical calculations
        - http_request: for API calls and web requests
        - file_read: for reading and analyzing files
        - python_repl: for executing Python code
        - google_search: for current information and web searches

        Think step by step and use tools as needed. 
        Explain your reasoning in the middle steps if it helps you decide.

        IMPORTANT:
        At the end of your reasoning, output the final answer **ONLY ONCE**. Do not include any explanation with the Final Answer.
        """

    # Create high-performance AgentWorkflowEngine with proper reset
    workflow_engine = AgentWorkflowEngine(
        workflow_cls=StrandsWorkflow,
        workflow_args={"agent_cls": StrandsAgent, "agent_args": {"model": model, "tools": tools, "system_prompt": system_prompt}},
        rollout_engine=rollout_engine,
        n_parallel_tasks=8,  # Enable parallel processing!
    )

    # Initialize the workflow pool
    await workflow_engine.initialize_pool()

    # Prepare task
    task = os.getenv(
        "STRANDS_TASK",
        "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?",
        # "What is 303 * 12314 - 123412 * 21234 + 128934 / 2910? Use calculator to solve this.",
    )

    task_dict = {"task": task}
    task_id = "strands_task_001"

    print(f"📝 Task: {task}")
    print(f"🔧 Available tools: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools]}")
    print("\n" + "=" * 80)
    print("🚀 Starting workflow execution...")
    print("=" * 80)

    # Execute with high-performance pooling
    episodes = await workflow_engine.execute_tasks([task_dict], [task_id])

    if not episodes:
        print("❌ No episodes returned from workflow")
        return

    episode = episodes[0]

    print("\n" + "=" * 80)
    print("✅ Workflow execution completed!")
    print("=" * 80)

    # Save episode to JSON file
    save_episode_to_json(episode)

    # Display concise episode summary
    print(f"\n✅ Episode {episode.id} completed")
    for agent_name, trajectory in episode.trajectories:
        print(f"📊 {agent_name}: {len(trajectory.steps)} steps, reward: {trajectory.reward}")


async def main():
    load_dotenv(find_dotenv())

    # Provider selection (Together or OpenAI-compatible)
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
