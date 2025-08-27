import asyncio
import os

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.terminal_litellm_engine import TerminalLiteLLMEngine
from rllm.workflows.terminal_workflow import TerminalWorkflow
from rllm.agents.terminal_terminus_agent import TerminalTerminusAgent
from rllm.environments.terminal.terminal_terminus import TerminalTerminusEnv
from examples.terminal.prepare_terminal_data import load_terminal_bench_dataset

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    dataset_name = "terminal-bench-core"
    dataset_version = "0.1.1"

    model_name = "openai/o4-mini"
    openai_base_url = None
    max_turns = 50
    max_agent_timeout_sec = 600.0

    env_args = {"model_name": model_name, "api_base": openai_base_url, "cleanup": True}
    rollout_engine = TerminalLiteLLMEngine(
        model=env_args["model_name"], api_base=env_args["api_base"]
    )

    engine = AgentWorkflowEngine(
        workflow_cls=TerminalWorkflow,
        workflow_args={
            "agent_cls": TerminalTerminusAgent,
            "env_cls": TerminalTerminusEnv,
            "env_args": env_args,
            "max_steps": max_turns,
            "global_agent_timeout_sec": max_agent_timeout_sec,
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=1,
        # Terminal-Bench already retries LLM calls 3 times in handle_llm_interaction
        retry_limit=1,
    )

    asyncio.run(engine.initialize_pool())

    tasks = load_terminal_bench_dataset(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    print(f"Loaded {len(tasks)} tasks from {dataset_name} {dataset_version}")

    episodes = asyncio.run(engine.execute_tasks(tasks=tasks))
    
    total = len(episodes)
    correct = sum(ep.is_correct for ep in episodes)
    print(f"Accuracy: {correct}/{total} = {correct / total:.3f}")