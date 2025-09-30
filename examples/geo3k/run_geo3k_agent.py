#!/usr/bin/env python3
"""
Inference script for GEO3K multimodal geometry agent.

This script tests the trained multimodal agent on GEO3K geometry problems
and evaluates its performance on problems with images.
"""

import asyncio
import os

from transformers import AutoTokenizer

from rllm.agents import Geo3kAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k


def main():
    """
    Main inference function for testing GEO3K multimodal agent.
    """
    print("🚀 GEO3K Multimodal Agent Inference")
    print("=" * 40)

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Configuration
    n_parallel_agents = 16  # Adjust based on your GPU memory
    model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Default multimodal model

    print(f"🔧 Configuration:")
    print(f"   - Model: {model_name}")
    print(f"   - Parallel agents: {n_parallel_agents}")

    # Initialize tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Agent configuration for multimodal geometry reasoning
    agent_args = {
        "accumulate_thinking": True,
        "include_images_in_completion": True,  # Include image info in completions for debugging
    }

    # Environment configuration
    env_args = {
        "reward_fn": math_reward_fn,
    }

    # Sampling parameters for inference
    sampling_params = {
        "temperature": 0.7,  # Slightly higher for reasoning diversity
        "top_p": 0.95,
        "max_tokens": 2048,
        "model": model_name
    }

    print("🔧 Initializing agent execution engine...")

    # Initialize execution engine
    # Note: You can switch between "openai" and "verl" engine
    # For multimodal models, "verl" engine with SGLang backend is recommended
    engine = AgentExecutionEngine(
        agent_class=Geo3kAgent,
        agent_args=agent_args,
        env_class=SingleTurnEnvironment,
        env_args=env_args,
        engine_name="openai",  # Can be "openai" or "verl"
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",  # SGLang server URL
            "api_key": "None"
        },
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=2048,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    # Load test dataset
    print("📊 Loading test dataset...")
    try:
        test_dataset = DatasetRegistry.load_dataset("geo3k_test", "test")
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    except Exception as e:
        print(f"❌ Dataset not found: {e}")
        print("🔄 Preparing dataset...")
        from prepare_geo3k_data import prepare_geo3k_data

        _, test_dataset = prepare_geo3k_data()

    test_data = test_dataset.get_data()

    # Take a smaller subset for quick testing
    test_samples = 50  # Adjust as needed
    subset_size = min(test_samples, len(test_data))
    test_subset = test_data[:subset_size]
    print(f"🎯 Testing on {subset_size} samples")

    # Check multimodal content statistics
    multimodal_count = sum(1 for sample in test_subset if sample.get("images"))
    print(f"📸 Multimodal samples: {multimodal_count}/{subset_size}")

    # Repeat samples for pass@k evaluation
    n_repeats = 4  # Number of attempts per problem
    tasks = [sample for sample in test_subset for _ in range(n_repeats)]
    print(f"🔄 Total tasks (with repeats): {len(tasks)}")

    # Show sample problem
    if not test_subset:
        print("⚠️ No samples found in test dataset. Exiting.")
        return

    sample = test_subset[0]
    print(f"\n📋 Sample Problem:")
    question_preview = sample.get("question", "")[:200]
    ground_truth_preview = str(sample.get("ground_truth", ""))[:100]
    has_images = bool(sample.get("images"))
    print(f"   Question: {question_preview}...")
    print(f"   Ground Truth: {ground_truth_preview}...")
    print(f"   Has Images: {has_images}")

    # Run inference
    print("\n🚀 Starting inference...")
    print("⏳ This may take a while depending on the number of samples and model speed...")

    try:
        results = asyncio.run(engine.execute_tasks(tasks))
        print(f"\n✅ Inference completed!")
        print(f"📊 Results: {len(results)} task completions")

        # Compute and display pass@k metrics
        print("\n📈 Computing Pass@K metrics...")
        pass_at_k_results = compute_pass_at_k(results)

        # Display results
        print(f"\n🎯 Performance Summary:")
        for k, score in pass_at_k_results.items():
            print(f"   - Pass@{k}: {score:.3f}")

        # Show some example results
        print(f"\n📝 Example Results:")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            reward = result.get('reward', 0)
            success = "✅" if reward > 0.5 else "❌"
            print(f"   {success} Task {i+1}: Reward = {reward:.3f}")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("💡 Make sure your model server is running:")
        print("   SGLang: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-2B-Instruct")
        print("   vLLM: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-VL-2B-Instruct")
        raise

    print(f"\n🎉 GEO3K inference completed!")
    print(f"💡 For better performance, consider:")
    print(f"   - Using a larger multimodal model (e.g., Qwen2-VL-7B)")
    print(f"   - Fine-tuning on GEO3K training data")
    print(f"   - Adjusting sampling parameters")


if __name__ == "__main__":
    main()
