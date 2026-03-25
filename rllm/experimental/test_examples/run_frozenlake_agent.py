import asyncio
import os
import sys

from transformers import AutoTokenizer

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.experimental.engine.agent_execution_workflow import AgentExecutionWorkflowEngine
from rllm.utils import compute_pass_at_k

_FROZENLAKE_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../../../examples/frozenlake")


def load_frozenlake_data():
    if DatasetRegistry.dataset_exists("frozenlake", "test"):
        test_dataset = DatasetRegistry.load_dataset("frozenlake", "test")
        return test_dataset.get_data()

    print("FrozenLake datasets not found. Preparing datasets...")
    sys.path.insert(0, _FROZENLAKE_EXAMPLES_DIR)
    from prepare_frozenlake_data import prepare_frozenlake_data

    train_dataset, test_dataset = prepare_frozenlake_data()

    return test_dataset.get_data()


if __name__ == "__main__":
    import tinker

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Hardcoded demo config for quick local runs.
    n_parallel_agents = 256

    model_name = "Qwen/Qwen3-8B"
    # Use None for default/local Tinker routing. Set to a URL if you have a dedicated Tinker endpoint.
    tinker_base_url = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 1.0, "top_p": 1.0}

    agent_args = {
        "max_steps": 10,
        "use_accumulate_history": True,
    }

    env_args = {
        "max_steps": 8,
        "is_slippery": False,
    }

    engine = AgentExecutionWorkflowEngine(
        agent_class=FrozenLakeAgent,
        env_class=FrozenLakeEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="tinker",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": tinker_base_url,
            "model_name": model_name,
            "tokenizer": tokenizer,
            "service_client": tinker.ServiceClient(base_url=tinker_base_url),
            "max_prompt_length": 4096,
            "max_response_length": 16384,
            "sampling_params": sampling_params,
        },
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )
    training_client = engine.rollout_engine.service_client.create_lora_training_client(
        base_model=model_name,
        rank=4,
    )
    sampler_path = training_client.save_weights_for_sampler(name="frozenlake_qwen3_8b_init_boredbichon").result().path
    sampling_client = training_client.create_sampling_client(sampler_path)
    engine.rollout_engine.set_sampling_client(sampling_client)

    tasks = load_frozenlake_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
