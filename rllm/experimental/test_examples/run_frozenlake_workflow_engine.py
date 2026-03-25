import asyncio
import os
import sys

import tinker
from transformers import AutoTokenizer

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.experimental.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.utils import compute_pass_at_k
from rllm.workflows.cumulative_workflow import CumulativeWorkflow

_FROZENLAKE_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../../../examples/frozenlake")


def load_frozenlake_data():
    if DatasetRegistry.dataset_exists("frozenlake", "test"):
        test_dataset = DatasetRegistry.load_dataset("frozenlake", "test")
        return test_dataset.get_data()

    print("FrozenLake datasets not found. Preparing datasets...")
    sys.path.insert(0, _FROZENLAKE_EXAMPLES_DIR)
    from prepare_frozenlake_data import prepare_frozenlake_data

    _, test_dataset = prepare_frozenlake_data()
    return test_dataset.get_data()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_tasks = 256
    model_name = "Qwen/Qwen3-8B"
    tinker_base_url = None
    sampling_params = {"temperature": 1.0, "top_p": 1.0}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    service_client = tinker.ServiceClient(base_url=tinker_base_url)

    rollout_engine = TinkerEngine(
        base_url=tinker_base_url,
        model_name=model_name,
        tokenizer=tokenizer,
        service_client=service_client,
        max_prompt_length=4096,
        max_response_length=16384,
        sampling_params=sampling_params,
        bypass_render_with_parser=True,
    )

    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=4,
    )
    sampler_path = training_client.save_weights_for_sampler(name="frozenlake_workflow_qwen3_8b_init_boredbichon").result().path
    sampling_client = training_client.create_sampling_client(sampler_path)
    rollout_engine.set_sampling_client(sampling_client)

    engine = AgentWorkflowEngine(
        workflow_cls=CumulativeWorkflow,
        workflow_args={
            "agent_cls": FrozenLakeAgent,
            "agent_args": {
                "max_steps": 10,
                "use_accumulate_history": True,
            },
            "env_cls": FrozenLakeEnv,
            "env_args": {
                "max_steps": 8,
                "is_slippery": False,
            },
            "max_steps": 10,
        },
        rollout_engine=rollout_engine,
        n_parallel_tasks=n_parallel_tasks,
        retry_limit=3,
    )

    tasks = load_frozenlake_data()
    episodes = asyncio.run(engine.execute_tasks(tasks))
    trajectories = [traj for ep in episodes for traj in ep.trajectories]
    compute_pass_at_k(trajectories)
    engine.shutdown()
