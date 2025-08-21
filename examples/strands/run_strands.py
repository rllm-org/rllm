import asyncio

from transformers import AutoTokenizer

from rllm.engine.rollout.openai_engine import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent

os.environ.setdefault("OTEL_SDK_DISABLED", "true")   # disable OTEL SDK


async def run_strands_agent(rollout_engine):
    """Example using StrandsAgent with trajectory tracking."""

    # Create RLLMModel
    model = RLLMModel(rollout_engine=rollout_engine, model_id="Qwen/Qwen3-0.6B")

    # Create StrandsAgent with trajectory tracking
    agent = StrandsAgent(model=model)

    # Reset trajectory for new task
    task = "Explain the concept of machine learning in simple terms"
    agent.reset_trajectory(task=task)

    print(f"📝 Task: {task}")

    # Run the agent
    result = agent("Explain the concept of machine learning in simple terms, using an analogy that a child could understand.")
    print(result)


async def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        base_url="http://localhost:30000/v1",
        api_key="None",
        sampling_params={"temperature": 0.7, "top_p": 0.95, "max_tokens": 512},
    )

    await run_strands_agent(rollout_engine)


if __name__ == "__main__":
    asyncio.run(main())
