import asyncio
import os

from transformers import AutoTokenizer

from projects.finqa.fin_qa_agent import FinQAAgent
from projects.finqa.fin_qa_environment import FinQAEnvironment
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import compute_pass_at_k

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = "rLLM/rLLM-FinQA-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95}

    engine = AgentExecutionEngine(
        agent_class=FinQAAgent,
        env_class=FinQAEnvironment,
        engine_name="openai",
        rollout_engine_args={"model": model_name, "base_url": "http://localhost:30000/v1", "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        n_parallel_agents=100,
        max_steps=20,
        max_prompt_length=4096,
    )

    test_dataset = DatasetRegistry.load_dataset("finqa", "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from projects.finqa.prepare_finqa_data import prepare_finqa_data

        _, _, test_dataset = prepare_finqa_data()

    tasks = test_dataset.repeat(n=1)

    results = asyncio.run(engine.execute_tasks(tasks))

    compute_pass_at_k(results)
