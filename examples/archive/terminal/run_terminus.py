import asyncio
import os

from prepare_terminal_data import load_terminal_bench_dataset
from terminus_workflow import TerminalTerminusWorkflow

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.openai_engine import OpenAIEngine


async def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Dataset selection (matches v0.2_tb style)
    dataset_name = "terminal-bench-core"
    dataset_version = "0.1.1"

    model_name = "o4-mini"
    rollout_engine = OpenAIEngine(model=model_name)

    max_steps = 50
    global_agent_timeout_sec = 600.0
    workflow_engine = AgentWorkflowEngine(
        workflow_cls=TerminalTerminusWorkflow,
        workflow_args={
            "model_name": model_name,
            "env_args": {
                "cleanup": True,
            },
            "max_steps": max_steps,
            "global_agent_timeout_sec": global_agent_timeout_sec,
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=1,
        retry_limit=1,  # TB already retries inside the agent loop
    )

    await workflow_engine.initialize_pool()

    # Load dataset
    tasks = load_terminal_bench_dataset(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )
    print(f"Loaded {len(tasks)} tasks from {dataset_name} {dataset_version}")

    # Execute all tasks
    episodes = await workflow_engine.execute_tasks(tasks=tasks)

    total = len(episodes)
    correct = sum(ep.is_correct for ep in episodes)
    print(f"Accuracy: {correct}/{total} = {correct / total:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
