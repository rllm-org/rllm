#!/usr/bin/env python3
"""
Concurrent evaluation script for BrowseComp-Plus dataset.
Hardcoded concurrency: 256
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.fully_async.client import RolloutClient

from .search_agent import rollout
from .tool_with_truncate import LocalRetrievalTool

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configuration
CONCURRENCY = 128
ROUTER_URL = "http://localhost:4000"
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_TURNS = 64
USE_REFINE = True

# URL file paths for auto-refresh support
VAL_RAG_URL_FILE = "/path/to/rllm/examples/fully_async/deep_research/.url/val_url"

# Validation uses browsecomp URL file with token-based truncation
val_retriever_tool = LocalRetrievalTool(
    url_file=VAL_RAG_URL_FILE,
    max_results=10,
    timeout=90.0,
    format_style="original",
)


async def evaluate_single_sample(client, sample, semaphore):
    """Evaluate a single sample with semaphore for concurrency control."""
    async with semaphore:
        try:
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            query_id = sample.get("query_id", "unknown")

            reward, metrics = await rollout(
                client=client,
                question=question,
                ground_truth=ground_truth,
                model=MODEL_NAME,
                max_turns=MAX_TURNS,
                use_refine=USE_REFINE,
                tool=val_retriever_tool,
            )

            # Add query_id to metrics
            metrics["query_id"] = query_id
            metrics["question"] = question
            metrics["ground_truth"] = ground_truth

            return metrics
        except Exception as e:
            print(f"Error evaluating sample {sample.get('query_id', 'unknown')}: {e}")
            import traceback

            traceback.print_exc()
            return {
                "query_id": sample.get("query_id", "unknown"),
                "error": str(e),
                "is_correct": False,
                "raw_reward": 0.0,
            }


async def evaluate_dataset(dataset, output_path: str = None):
    """Evaluate entire dataset with concurrent execution."""
    print(f"[*] Starting evaluation on {len(dataset)} samples")
    print(f"[*] Concurrency: {CONCURRENCY}")
    print(f"[*] Router URL: {ROUTER_URL}")
    print(f"[*] Model: {MODEL_NAME}")
    print(f"[*] Max turns: {MAX_TURNS}")
    print(f"[*] Use refine: {USE_REFINE}")
    print()

    # Initialize client
    client = RolloutClient(
        router_url=ROUTER_URL,
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        max_tokens=40_000,
    )

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Create tasks for all samples
    tasks = [evaluate_single_sample(client, sample, semaphore) for sample in dataset]

    # Run with progress bar that updates with running statistics
    start_time = time.time()
    results = []

    from tqdm.asyncio import tqdm

    pbar = tqdm(total=len(tasks), desc="Evaluating")

    # Process results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

        # Calculate running statistics
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            f1_scores = [r.get("f1_score", 0.0) for r in valid_results]
            exact_matches = [r.get("exact_match", False) for r in valid_results]
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_em = sum(exact_matches) / len(exact_matches)

            pbar.set_postfix({"avg_f1": f"{avg_f1:.4f}", "avg_em": f"{avg_em:.2%}", "valid": len(valid_results)})

        pbar.update(1)

    pbar.close()
    total_time = time.time() - start_time

    # Compute statistics
    stats = compute_statistics(results, total_time)

    # Print statistics
    print_statistics(stats)

    # Save results if output path provided
    if output_path:
        save_results(results, stats, output_path)

    return results, stats


def compute_statistics(results, total_time):
    """Compute evaluation statistics from results."""
    stats = {
        "total_samples": len(results),
        "total_time": total_time,
        "samples_per_second": len(results) / total_time if total_time > 0 else 0,
    }

    # Filter out error results
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    stats["num_errors"] = len(error_results)
    stats["num_valid"] = len(valid_results)

    if not valid_results:
        print("[!] No valid results to compute statistics")
        return stats

    # Accuracy metrics
    correct_results = [r for r in valid_results if r.get("is_correct", False)]
    stats["num_correct"] = len(correct_results)
    stats["accuracy"] = len(correct_results) / len(valid_results)

    # F1 and Exact Match
    f1_scores = [r.get("f1_score", 0.0) for r in valid_results]
    exact_matches = [r.get("exact_match", False) for r in valid_results]

    stats["avg_f1"] = sum(f1_scores) / len(f1_scores)
    stats["avg_exact_match"] = sum(exact_matches) / len(exact_matches)
    stats["median_f1"] = sorted(f1_scores)[len(f1_scores) // 2]

    # Precision and Recall
    precisions = [r.get("precision", 0.0) for r in valid_results]
    recalls = [r.get("recall", 0.0) for r in valid_results]
    stats["avg_precision"] = sum(precisions) / len(precisions)
    stats["avg_recall"] = sum(recalls) / len(recalls)

    # Tool call statistics
    tool_calls = [r.get("total_tool_calls", 0) for r in valid_results]
    stats["avg_tool_calls"] = sum(tool_calls) / len(tool_calls)
    stats["median_tool_calls"] = sorted(tool_calls)[len(tool_calls) // 2]
    stats["max_tool_calls"] = max(tool_calls) if tool_calls else 0
    stats["min_tool_calls"] = min(tool_calls) if tool_calls else 0

    # Turn statistics
    num_turns = [r.get("num_turns", 0) for r in valid_results]
    stats["avg_turns"] = sum(num_turns) / len(num_turns)
    stats["median_turns"] = sorted(num_turns)[len(num_turns) // 2]

    # Token statistics
    completion_tokens = [r.get("total_completion_tokens", 0) for r in valid_results]
    tool_tokens = [r.get("total_tool_tokens", 0) for r in valid_results]
    stats["avg_completion_tokens"] = sum(completion_tokens) / len(completion_tokens)
    stats["avg_tool_tokens"] = sum(tool_tokens) / len(tool_tokens)
    stats["total_completion_tokens"] = sum(completion_tokens)
    stats["total_tool_tokens"] = sum(tool_tokens)

    # Time statistics
    gen_times = [r.get("total_generation_time", 0) for r in valid_results]
    tool_wait_times = [r.get("total_tool_wait_time", 0) for r in valid_results]
    refine_times = [r.get("total_refine_time", 0) for r in valid_results]

    stats["avg_generation_time"] = sum(gen_times) / len(gen_times)
    stats["avg_tool_wait_time"] = sum(tool_wait_times) / len(tool_wait_times)
    stats["avg_refine_time"] = sum(refine_times) / len(refine_times)

    # Error detection statistics
    stats["duplicate_search_rate"] = sum(r.get("duplicate_search_detected", False) for r in valid_results) / len(valid_results)
    stats["excessive_parallel_calls_rate"] = sum(r.get("excessive_parallel_calls", False) for r in valid_results) / len(valid_results)
    stats["tool_error_rate"] = sum(r.get("tool_error_detected", False) for r in valid_results) / len(valid_results)
    stats["refine_error_rate"] = sum(r.get("refine_error_detected", False) for r in valid_results) / len(valid_results)
    stats["overlong_rate"] = sum(r.get("overlong", False) for r in valid_results) / len(valid_results)

    # Reward statistics
    rewards = [r.get("raw_reward", 0.0) for r in valid_results]
    stats["avg_reward"] = sum(rewards) / len(rewards)
    stats["median_reward"] = sorted(rewards)[len(rewards) // 2]

    return stats


def print_statistics(stats):
    """Print formatted statistics."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\n📊 Overall Statistics:")
    print(f"  Total samples:        {stats['total_samples']}")
    print(f"  Valid results:        {stats['num_valid']}")
    print(f"  Errors:               {stats['num_errors']}")
    print(f"  Total time:           {stats['total_time']:.2f}s")
    print(f"  Throughput:           {stats['samples_per_second']:.2f} samples/s")

    if stats["num_valid"] == 0:
        return

    print("\n🎯 Accuracy Metrics:")
    print(f"  Accuracy:             {stats['accuracy']:.2%}")
    print(f"  Correct:              {stats['num_correct']}/{stats['num_valid']}")
    print(f"  Avg F1 Score:         {stats['avg_f1']:.4f}")
    print(f"  Median F1 Score:      {stats['median_f1']:.4f}")
    print(f"  Avg Exact Match:      {stats['avg_exact_match']:.2%}")
    print(f"  Avg Precision:        {stats['avg_precision']:.4f}")
    print(f"  Avg Recall:           {stats['avg_recall']:.4f}")

    print("\n🔧 Tool Usage Statistics:")
    print(f"  Avg tool calls:       {stats['avg_tool_calls']:.2f}")
    print(f"  Median tool calls:    {stats['median_tool_calls']}")
    print(f"  Min/Max tool calls:   {stats['min_tool_calls']}/{stats['max_tool_calls']}")
    print(f"  Avg turns:            {stats['avg_turns']:.2f}")
    print(f"  Median turns:         {stats['median_turns']}")

    print("\n💬 Token Statistics:")
    print(f"  Avg completion tokens: {stats['avg_completion_tokens']:.0f}")
    print(f"  Avg tool tokens:       {stats['avg_tool_tokens']:.0f}")
    print(f"  Total completion tokens: {stats['total_completion_tokens']:,}")
    print(f"  Total tool tokens:     {stats['total_tool_tokens']:,}")

    print("\n⏱️  Time Statistics:")
    print(f"  Avg generation time:  {stats['avg_generation_time']:.2f}s")
    print(f"  Avg tool wait time:   {stats['avg_tool_wait_time']:.2f}s")
    print(f"  Avg refine time:      {stats['avg_refine_time']:.2f}s")

    print("\n⚠️  Error Rates:")
    print(f"  Duplicate search:     {stats['duplicate_search_rate']:.2%}")
    print(f"  Excessive parallel:   {stats['excessive_parallel_calls_rate']:.2%}")
    print(f"  Tool errors:          {stats['tool_error_rate']:.2%}")
    print(f"  Refine errors:        {stats['refine_error_rate']:.2%}")
    print(f"  Overlong:             {stats['overlong_rate']:.2%}")

    print("\n🏆 Reward Statistics:")
    print(f"  Avg reward:           {stats['avg_reward']:.4f}")
    print(f"  Median reward:        {stats['median_reward']:.4f}")

    print("=" * 80 + "\n")


def save_results(results, stats, output_path: str):
    """Save results and statistics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for JSON serialization
    # Remove non-serializable objects (trajectory, messages)
    serializable_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k not in ["trajectory", "messages"]}
        serializable_results.append(r_copy)

    output_data = {
        "statistics": stats,
        "results": serializable_results,
        "config": {
            "concurrency": CONCURRENCY,
            "router_url": ROUTER_URL,
            "model_name": MODEL_NAME,
            "max_turns": MAX_TURNS,
            "use_refine": USE_REFINE,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[*] Results saved to: {output_path}")


async def main():
    """Main entry point."""
    # Load dataset
    print("[*] Loading BrowseComp-Plus dataset...")
    dataset = DatasetRegistry.load_dataset("browsecomp-plus", split="test")
    print(f"[*] Loaded {len(dataset)} samples\n")

    # Run evaluation
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"eval_results_browsecomp_{timestamp}.json"

    results, stats = await evaluate_dataset(dataset, output_path=output_path)

    return results, stats


if __name__ == "__main__":
    asyncio.run(main())
