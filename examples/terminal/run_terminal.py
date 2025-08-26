import asyncio
import os
from pathlib import Path


from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.terminal_litellm_engine import TerminalLiteLLMEngine
# from rllm.parser.chat_template.parser import ChatTemplateParser
from rllm.workflows.terminal_workflow import TerminalWorkflow
from rllm.agents.terminal_terminus_agent import TerminalTerminusAgent
from rllm.environments.terminal.terminal_terminus import TerminalTerminusEnv
# from rllm.registry.dataset_registry import DatasetRegistry
from examples.terminal.prepare_terminal_data import load_terminal_bench_dataset

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Terminal-Bench dataset selection
    DATASET_NAME = "terminal-bench-core"
    DATASET_VERSION = "0.1.1"

    # Model/backend config (adjust as needed)
    MODEL_NAME = "openai/o4-mini"  # Match Terminal-Bench model naming
    OPENAI_BASE_URL = None
    MAX_TURNS = 50

    # Rollout engine that uses Terminal-Bench's LiteLLM for exact parity
    rollout_engine = TerminalLiteLLMEngine(model=MODEL_NAME, api_base=OPENAI_BASE_URL)

    # Workflow engine with MultiTurnWorkflow
    engine = AgentWorkflowEngine(
        workflow_cls=TerminalWorkflow,
        workflow_args={
            "agent_cls": TerminalTerminusAgent,
            "env_cls": TerminalTerminusEnv,
            "agent_args": {},
            "env_args": {"model_name": MODEL_NAME, "api_base": OPENAI_BASE_URL, "cleanup": True},
            "max_steps": MAX_TURNS,
            "sampling_params": {
                "model": MODEL_NAME,
            },
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=2,
        retry_limit=3,
    )


    asyncio.run(engine.initialize_pool())

    # Load all tasks from terminal-bench-core v0.1.1 using remote registry
    tasks = load_terminal_bench_dataset(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
    )[:5]

    print(f"Loaded {len(tasks)} tasks from {DATASET_NAME} {DATASET_VERSION}")

    # Run multiple rollouts per task by duplicating the task list and passing matching task_ids
    NUM_ROLLOUTS_PER_TASK = 2  # adjust as needed
    tasks_to_run = [t for t in tasks for _ in range(NUM_ROLLOUTS_PER_TASK)]
    task_ids = [t.get("task_id", str(i)) for i, t in enumerate(tasks) for _ in range(NUM_ROLLOUTS_PER_TASK)]

    episodes = asyncio.run(engine.execute_tasks(tasks=tasks_to_run, task_ids=task_ids))

    total = 0
    correct = 0
    for ep in episodes:
        total += 1
        correct += ep.is_correct
        with open("tb-results.txt", "a") as f:
            f.write(f"Episode ID: {ep.id}\n")
            f.write(f"Task: {ep.task}\n")
            for name, traj in ep.trajectories:
                f.write(f"Trajectory {name}: steps={len(traj.steps)} reward={traj.reward}\n")
            f.write(f"Is Correct: {ep.is_correct}\n")
            f.write(f"Termination: {ep.termination_reason}\n")
            f.write(f"total: {total}\n")
            f.write(f"correct: {correct}\n")
            f.write(f"accuracy: {correct / total if total else 0.0}\n")
    print(f"Total episodes: {total} (expected {len(tasks_to_run)})")
    print(f"Correct episodes: {correct}")
    print(f"Accuracy: {correct / total if total else 0.0}")