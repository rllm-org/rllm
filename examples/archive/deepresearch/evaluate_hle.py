"""
Humanity's Last Exam (HLE) Evaluation for DeepResearch + rLLM

Adapted from original DeepResearch HLE evaluation to work with rLLM's
DeepResearch integration and AgentWorkflowEngine.

Original: https://github.com/Alibaba-NLP/DeepResearch/blob/main/evaluation/evaluate_hle_official.py
"""

import argparse
import asyncio
import json
import os
import statistics
from datetime import datetime
from typing import Any

from datasets import load_dataset
from deepresearch_tools import get_all_tools
from deepresearch_workflow import DeepResearchWorkflow
from dotenv import find_dotenv, load_dotenv

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout import OpenAIEngine


class HLEJudge:
    """Judge for evaluating HLE responses using OpenAI API."""

    def __init__(self, judge_engine: OpenAIEngine):
        self.judge_engine = judge_engine
        # Binary yes/no judge prompt aligned with Tongyi DeepResearch
        self.judge_prompt = """You are an impartial judge evaluating the correctness of an AI assistant's answer.

[Question]
{question}

[Correct Answer]
{reference_answer}

[Assistant's Answer]
{assistant_answer}

Task: Determine if the assistant's answer is correct by comparing it to the correct answer.

Instructions:
1. Extract the final answer from the assistant's response
2. Compare it with the correct answer
3. Provide your reasoning
4. Answer with "yes" if correct, "no" if incorrect

Output format:
correct: [yes/no]
reasoning: [your explanation]"""

    async def judge_response(self, question: str, reference_answer: str, assistant_answer: str) -> dict[str, Any]:
        """
        Judge a single response.

        Args:
            question: Original question
            reference_answer: Ground truth answer
            assistant_answer: Model's prediction

        Returns:
            Dictionary with judgment results
        """
        try:
            prompt = self.judge_prompt.format(
                question=question,
                reference_answer=reference_answer,
                assistant_answer=assistant_answer,
            )

            messages = [{"role": "user", "content": prompt}]

            # Use appropriate token parameter based on model
            if "o3" in self.judge_engine.model.lower() or "o1" in self.judge_engine.model.lower():
                response = await self.judge_engine.get_model_response(messages=messages, max_completion_tokens=1000)
            else:
                response = await self.judge_engine.get_model_response(messages=messages, temperature=0.1, max_tokens=1000)

            judgment_text = response.text if hasattr(response, "text") else str(response)

            # Parse binary yes/no from judge output
            is_correct = False
            if "correct:" in judgment_text.lower():
                # Extract the yes/no after "correct:"
                try:
                    correct_line = [line for line in judgment_text.lower().split("\n") if "correct:" in line][0]
                    is_correct = "yes" in correct_line
                except (IndexError, ValueError):
                    is_correct = False

            return {
                "judgment": judgment_text,
                "is_correct": is_correct,
            }

        except Exception as e:
            print(f"Judge error: {e}")
            return {"judgment": f"Judge error: {e}", "is_correct": False}


async def evaluate_hle_dataset(dataset_path: str, args) -> dict[str, Any]:
    """
    Evaluate DeepResearch on HLE dataset.

    Args:
        dataset_path: Path to HLE JSONL dataset
        args: Command line arguments

    Returns:
        Evaluation results dictionary
    """
    print("📊 Starting HLE Evaluation")
    print(f"Dataset: {dataset_path}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 60)

    # Load dataset (HF only to align with examples pattern)
    questions = []
    dataset_name = args.hf_dataset or "cais/hle"
    split_name = args.hf_split or "test"

    print(f"🧰 Loading dataset from Hugging Face: {dataset_name} (split={split_name})")
    try:
        if args.hf_config:
            ds = load_dataset(dataset_name, args.hf_config, split=split_name)
        else:
            ds = load_dataset(dataset_name, split=split_name)

        def extract_qa(example: dict[str, Any]) -> dict[str, str]:
            q = ""
            a = ""
            if "question" in example:
                q = example["question"]
            elif "prompt" in example:
                q = example["prompt"]
            elif "input" in example:
                q = example["input"]

            if "answer" in example:
                a = example["answer"]
            elif "target" in example:
                a = example["target"]
            elif "output" in example:
                a = example["output"]
            elif "correct_answer" in example:
                a = example["correct_answer"]

            if "choices" in example and a:
                try:
                    choices_text = "\n".join([f"{i + 1}. {choice}" for i, choice in enumerate(example["choices"])])
                    q = f"{q}\n\nChoices:\n{choices_text}"
                except Exception:
                    pass

            # Inject external contexts (urls/files/images/extra text) to help tools
            try:
                extras: list[str] = []
                # Text contexts
                for key in [
                    "context",
                    "contexts",
                    "extra",
                    "additional_context",
                    "background",
                    "passage",
                    "passages",
                ]:
                    if key in example and example[key]:
                        val = example[key]
                        if isinstance(val, list | tuple):
                            val_str = "\n".join([str(v) for v in val][:5])
                        else:
                            val_str = str(val)
                        if val_str.strip():
                            extras.append(f"{key.title()}:\n{val_str}")

                # URLs
                urls = []
                if "urls" in example and example["urls"]:
                    urls = example["urls"] if isinstance(example["urls"], list | tuple) else [example["urls"]]
                elif "url" in example and example["url"]:
                    urls = [example["url"]]
                if urls:
                    url_lines = "\n".join([f"- {u}" for u in urls[:10]])
                    extras.append(f"URLs:\n{url_lines}")

                # File paths
                file_paths = []
                for key in ["file_paths", "file_path", "files"]:
                    if key in example and example[key]:
                        vals = example[key] if isinstance(example[key], list | tuple) else [example[key]]
                        file_paths.extend([str(v) for v in vals])
                if file_paths:
                    file_lines = "\n".join([f"- {p}" for p in file_paths[:10]])
                    extras.append(f"Files:\n{file_lines}")

                # Images
                images = []
                for key in ["images", "image"]:
                    if key in example and example[key]:
                        vals = example[key] if isinstance(example[key], list | tuple) else [example[key]]
                        images.extend([str(v) for v in vals])
                if images:
                    img_lines = "\n".join([f"- {p}" for p in images[:10]])
                    extras.append(f"Images:\n{img_lines}")

                if extras:
                    q = f"{q}\n\nAdditional context for tools:\n" + "\n\n".join(extras)
            except Exception:
                pass

            return {
                "question": str(q) if q is not None else "",
                "answer": str(a) if a is not None else "",
            }

        total_len = len(ds)
        limit = min(args.max_samples, total_len) if args.max_samples else total_len
        for idx in range(limit):
            ex = ds[idx]
            qa = extract_qa(ex)
            if qa["question"] and qa["answer"]:
                questions.append(
                    {
                        "id": f"hle_{idx}",
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )
            else:
                print(f"Warning: Could not extract question/answer from example {idx}")

    except Exception as e:
        print(f"❌ Failed to load dataset from Hugging Face: {e}")
        raise

    print(f"📋 Loaded {len(questions)} questions from HLE dataset")

    # Setup rollout engine
    load_dotenv(find_dotenv())

    # Use GPT-4o for model evaluation
    model_engine = setup_rollout_engine(args, model_role="evaluation")

    # Setup judge (can use same or different model)
    judge_engine = setup_rollout_engine(args, model_role="judge")
    judge = HLEJudge(judge_engine)

    # Setup tools
    tools = get_all_tools()

    # Create AgentWorkflowEngine
    workflow_engine = AgentWorkflowEngine(
        workflow_cls=DeepResearchWorkflow,
        workflow_args={
            "tools": tools,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        rollout_engine=model_engine,
        n_parallel_tasks=args.parallel_tasks,
        retry_limit=1,
    )

    print(f"⚙️  Created evaluation setup with {args.parallel_tasks} parallel tasks")

    # Run DeepResearch evaluation
    print("\n🔬 Running DeepResearch evaluation...")
    start_time = asyncio.get_event_loop().time()

    try:
        episodes = await workflow_engine.execute_tasks(questions)
        eval_time = asyncio.get_event_loop().time() - start_time

        print(f"\n✅ Evaluation completed in {eval_time:.1f}s")

        # Extract predictions
        results = []
        for episode in episodes:
            prediction = episode.metrics.get("prediction", "No prediction available")
            results.append(
                {
                    "question": episode.task.get("question", ""),
                    "reference_answer": episode.task.get("answer", ""),
                    "prediction": prediction,
                    "episode_id": episode.id,
                    "is_correct": episode.is_correct,
                    "rounds": episode.metrics.get("rounds", 0),
                    "termination_reason": episode.termination_reason.value if episode.termination_reason else "unknown",
                }
            )

        # Judge responses
        print(f"\n⚖️  Judging {len(results)} responses...")

        judge_results = []
        for result in results:
            judgment = await judge.judge_response(
                question=result["question"],
                reference_answer=result["reference_answer"],
                assistant_answer=result["prediction"],
            )
            result.update(judgment)
            judge_results.append(result)

        # Calculate metrics
        metrics = calculate_hle_metrics(judge_results)
        metrics["evaluation_time"] = eval_time
        metrics["total_questions"] = len(questions)

        # Save results
        save_hle_results(judge_results, metrics, args)

        return metrics

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


def setup_rollout_engine(args, model_role="evaluation") -> OpenAIEngine:
    """Setup rollout engine for evaluation or judging."""

    # Load environment variables
    load_dotenv(find_dotenv())

    # Provider selection
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if args.api_key:
        api_key = args.api_key
        base_url = args.base_url or "https://api.openai.com/v1"
        model_name = args.model or "gpt-4"
    elif together_api_key and model_role == "evaluation":
        api_key = together_api_key
        base_url = args.base_url or "https://api.together.xyz/v1"
        model_name = args.model or os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
        print(f"🔧 Using Together AI for {model_role}")
    elif openai_api_key:
        api_key = openai_api_key
        base_url = args.base_url or "https://api.openai.com/v1"
        model_name = args.model or "gpt-4o"
        print(f"🔧 Using OpenAI for {model_role}")
    else:
        raise ValueError("❌ API key required. Please set OPENAI_API_KEY or TOGETHER_AI_API_KEY in .env file")

    # For evaluation, DeepResearch handles all sampling params internally
    # For judge, we need basic params
    if model_role == "judge":
        # Check if model is O3/O1 (use model_name which is already determined above)
        if "o3" in model_name.lower() or "o1" in model_name.lower():
            sampling_params = {
                "max_completion_tokens": 1000,
            }
        else:
            sampling_params = {
                "temperature": 0.1,
                "top_p": 0.95,
                "max_tokens": 1000,
            }
    else:
        # Don't set default sampling_params for evaluation
        # DeepResearch will handle model-specific params
        sampling_params = {}

    return OpenAIEngine(
        model=model_name,
        tokenizer=None,
        base_url=base_url,
        api_key=api_key,
        sampling_params=sampling_params,
    )


def calculate_hle_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate HLE evaluation metrics."""

    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}

    # Basic accuracy (judge-based binary yes/no)
    judge_correct = sum(1 for r in results if r.get("is_correct", False))
    judge_accuracy = judge_correct / total

    # Termination analysis
    termination_counts = {}
    for result in results:
        reason = result.get("termination_reason", "unknown")
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    # Round analysis
    rounds = [r.get("rounds", 0) for r in results]
    avg_rounds = statistics.mean(rounds) if rounds else 0

    return {
        "total_questions": total,
        "judge_accuracy": judge_accuracy,
        "judge_correct": judge_correct,
        "average_rounds": avg_rounds,
        "termination_distribution": termination_counts,
    }


def save_hle_results(results: list[dict], metrics: dict, args):
    """Save HLE evaluation results."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = os.path.join(args.output_dir, f"hle_results_{timestamp}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "dataset": "HLE",
                    "model": args.model,
                    "total_questions": len(results),
                },
                "results": results,
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save metrics summary
    metrics_file = os.path.join(args.output_dir, f"hle_metrics_{timestamp}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"💾 Results saved to: {results_file}")
    print(f"📊 Metrics saved to: {metrics_file}")


def print_hle_summary(metrics: dict[str, Any]):
    """Print HLE evaluation summary."""

    print("\n" + "=" * 60)
    print("📊 HLE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions: {metrics.get('total_questions', 0)}")
    print(f"Judge Accuracy: {metrics.get('judge_accuracy', 0):.2%}")
    print(f"Correct Answers: {metrics.get('judge_correct', 0)}/{metrics.get('total_questions', 0)}")
    print(f"Average Rounds: {metrics.get('average_rounds', 0):.1f}")
    print(f"Evaluation Time: {metrics.get('evaluation_time', 0):.1f}s")

    print("\nTermination Reasons:")
    term_dist = metrics.get("termination_distribution", {})
    for reason, count in term_dist.items():
        print(f"  {reason}: {count}")

    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Run HLE evaluation with DeepResearch + rLLM")

    # Dataset options (HF only)
    parser.add_argument(
        "--hf-dataset",
        default="cais/hle",
        help="Hugging Face dataset path (default: cais/hle)",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Optional dataset configuration name for HF datasets that require it.",
    )
    parser.add_argument(
        "--hf-split",
        default="test",
        help="Dataset split to load from HF (default: test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )

    # Model options
    parser.add_argument("--model", default=None, help="Model name to use")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key (uses env vars if not provided)")

    # Execution options
    parser.add_argument("--parallel-tasks", type=int, default=4, help="Number of parallel tasks")
    parser.add_argument("--output-dir", default="./hle_outputs", help="Output directory for results")

    args = parser.parse_args()

    try:
        metrics = await evaluate_hle_dataset(args.hf_dataset, args)
        print_hle_summary(metrics)

    except Exception as e:
        print(f"❌ HLE evaluation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Set environment for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    asyncio.run(main())
