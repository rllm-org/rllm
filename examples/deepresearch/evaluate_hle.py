"""
Humanity's Last Exam (HLE) Evaluation for DeepResearch + rLLM

Adapted from original DeepResearch HLE evaluation to work with rLLM's
DeepResearch integration and AgentWorkflowEngine.

Original: https://github.com/Alibaba-NLP/DeepResearch/blob/main/evaluation/evaluate_hle_official.py

Evaluation Method:
- Uses o3-mini as judge model (aligned with Tongyi's official evaluation)
- Binary yes/no judgment with structured output (Pydantic schema)
- Strict matching based on [correct_answer] with small numerical tolerance
- Final metric: accuracy (0-100%) computed as correct/total
"""

import asyncio
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any, Literal
import statistics

from dotenv import find_dotenv, load_dotenv
from datasets import load_dataset
from pydantic import BaseModel

from rllm.engine.rollout import OpenAIEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from deepresearch_workflow import DeepResearchWorkflow
from deepresearch_tools import get_all_tools


# Pydantic schema for structured judge output (aligned with Tongyi)
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int


class HLEJudge:
    """
    Judge for evaluating HLE responses using o3-mini with structured output.

    Aligned with Tongyi's official evaluation method:
    https://github.com/Alibaba-NLP/DeepResearch/blob/main/evaluation/evaluate_hle_official.py
    """

    def __init__(self, judge_engine: OpenAIEngine):
        self.judge_engine = judge_engine
        # Tongyi's original judge prompt (binary yes/no with strict matching)
        self.judge_prompt = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

    async def judge_response(
        self, question: str, reference_answer: str, assistant_answer: str
    ) -> Dict[str, Any]:
        """
        Judge a single response using structured output.

        Args:
            question: Original question
            reference_answer: Ground truth answer
            assistant_answer: Model's prediction

        Returns:
            Dictionary with judgment results (aligned with Tongyi format)
        """
        try:
            prompt = self.judge_prompt.format(
                question=question,
                correct_answer=reference_answer,
                response=assistant_answer,
            )

            # Add explicit JSON format instruction (required for OpenAI JSON mode)
            prompt += "\n\nPlease respond in JSON format with the following fields: extracted_final_answer, reasoning, correct, confidence."

            messages = [{"role": "user", "content": prompt}]

            # Use JSON mode for structured output (compatible with o3-mini)
            response = await self.judge_engine.get_model_response(
                messages=messages,
                max_completion_tokens=8192,
                response_format={"type": "json_object"},
            )

            judgment_text = (
                response.text if hasattr(response, "text") else str(response)
            )

            # Parse structured JSON output
            try:
                judgment_data = json.loads(judgment_text)
                extracted_answer = judgment_data.get("extracted_final_answer", "None")
                reasoning = judgment_data.get("reasoning", "")
                correct = judgment_data.get("correct", "no")
                confidence = judgment_data.get("confidence", 0)
            except json.JSONDecodeError:
                # Fallback: try to extract from text
                print("⚠️  Failed to parse JSON, using text fallback")
                extracted_answer = "None"
                reasoning = judgment_text
                correct = "yes" if "correct: yes" in judgment_text.lower() else "no"
                confidence = 0

            # Binary judgment: yes/no
            is_correct = correct.lower() == "yes"

            return {
                "judgment": reasoning,
                "extracted_answer": extracted_answer,
                "correct": correct,
                "confidence": confidence,
                "is_correct": is_correct,
                "rating": 5 if is_correct else 1,  # For compatibility with old metrics
            }

        except Exception as e:
            print(f"Judge error: {e}")
            return {
                "judgment": f"Judge error: {e}",
                "extracted_answer": "None",
                "correct": "no",
                "confidence": 0,
                "is_correct": False,
                "rating": 0,
            }


async def evaluate_hle_dataset(dataset_path: str, args) -> Dict[str, Any]:
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

        def extract_qa(example: Dict[str, Any]) -> Dict[str, str]:
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
                    choices_text = "\n".join(
                        [
                            f"{i + 1}. {choice}"
                            for i, choice in enumerate(example["choices"])
                        ]
                    )
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
                        if isinstance(val, (list, tuple)):
                            val_str = "\n".join([str(v) for v in val][:5])
                        else:
                            val_str = str(val)
                        if val_str.strip():
                            extras.append(f"{key.title()}:\n{val_str}")

                # URLs
                urls = []
                if "urls" in example and example["urls"]:
                    urls = (
                        example["urls"]
                        if isinstance(example["urls"], (list, tuple))
                        else [example["urls"]]
                    )
                elif "url" in example and example["url"]:
                    urls = [example["url"]]
                if urls:
                    url_lines = "\n".join([f"- {u}" for u in urls[:10]])
                    extras.append(f"URLs:\n{url_lines}")

                # File paths
                file_paths = []
                for key in ["file_paths", "file_path", "files"]:
                    if key in example and example[key]:
                        vals = (
                            example[key]
                            if isinstance(example[key], (list, tuple))
                            else [example[key]]
                        )
                        file_paths.extend([str(v) for v in vals])
                if file_paths:
                    file_lines = "\n".join([f"- {p}" for p in file_paths[:10]])
                    extras.append(f"Files:\n{file_lines}")

                # Images - Store for multi-modal message construction
                images = []
                for key in ["images", "image"]:
                    if key in example and example[key]:
                        vals = (
                            example[key]
                            if isinstance(example[key], (list, tuple))
                            else [example[key]]
                        )
                        images.extend([str(v) for v in vals])

                # Store images for vision model processing
                # Note: Images will be sent directly to vision model via multimodal messages

                if extras:
                    q = f"{q}\n\nAdditional context for tools:\n" + "\n\n".join(extras)
            except Exception:
                pass

            result = {
                "question": str(q) if q is not None else "",
                "answer": str(a) if a is not None else "",
            }

            # Include images if present
            if images:
                result["_images"] = images

            return result

        total_len = len(ds)
        limit = min(args.max_samples, total_len) if args.max_samples else total_len
        for idx in range(limit):
            ex = ds[idx]
            qa = extract_qa(ex)
            if qa["question"] and qa["answer"]:
                task = {
                    "id": f"hle_{idx}",
                    "question": qa["question"],
                    "answer": qa["answer"],
                }
                # Include images if present
                if "_images" in qa:
                    task["_images"] = qa["_images"]
                questions.append(task)
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
                    "termination_reason": episode.termination_reason.value
                    if episode.termination_reason
                    else "unknown",
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
    """
    Setup rollout engine for evaluation or judging.

    For judge: defaults to o3-mini (aligned with Tongyi's official evaluation)
    For evaluation: defaults to gpt-4o or Together AI model
    """

    # Load environment variables
    load_dotenv(find_dotenv())

    # Provider selection
    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if args.api_key:
        api_key = args.api_key
        base_url = args.base_url or "https://api.openai.com/v1"
        if model_role == "judge":
            model_name = args.judge_model or "o3-mini"  # Tongyi's default
        else:
            model_name = args.model or "gpt-4o"
    elif together_api_key and model_role == "evaluation":
        api_key = together_api_key
        base_url = args.base_url or "https://api.together.xyz/v1"
        model_name = args.model or os.getenv(
            "TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo"
        )
        print(f"🔧 Using Together AI for {model_role}")
    elif openai_api_key:
        api_key = openai_api_key
        base_url = args.base_url or "https://api.openai.com/v1"
        if model_role == "judge":
            model_name = args.judge_model if hasattr(args, "judge_model") else "o3-mini"
            print(f"🔧 Using {model_name} for {model_role} (Tongyi-aligned)")
        else:
            model_name = args.model or "gpt-4o"
            print(f"🔧 Using OpenAI for {model_role}")
    else:
        raise ValueError(
            "❌ API key required. Please set OPENAI_API_KEY or TOGETHER_AI_API_KEY in .env file"
        )

    # Judge uses simpler sampling params
    if model_role == "judge":
        # For o3-mini, directly use max_completion_tokens to avoid warnings
        if model_name and model_name.lower().startswith("o3"):
            sampling_params = {
                "max_completion_tokens": 8192,
            }
        else:
            sampling_params = {
                "max_tokens": 8192,
            }
    else:
        sampling_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 2048,
        }

    return OpenAIEngine(
        model=model_name,
        tokenizer=None,
        base_url=base_url,
        api_key=api_key,
        sampling_params=sampling_params,
    )


def calculate_hle_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate HLE evaluation metrics."""

    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}

    # Basic accuracy (judge-based)
    judge_correct = sum(1 for r in results if r.get("is_correct", False))
    judge_accuracy = judge_correct / total

    # Confidence distribution (from judge)
    confidences = [r.get("confidence", 0) for r in results if "confidence" in r]
    avg_confidence = statistics.mean(confidences) if confidences else 0

    # Termination analysis
    termination_counts = {}
    for result in results:
        reason = result.get("termination_reason", "unknown")
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    # Round analysis
    rounds = [r.get("rounds", 0) for r in results]
    avg_rounds = statistics.mean(rounds) if rounds else 0

    # Judgment distribution (yes/no)
    correct_judgments = sum(1 for r in results if r.get("correct") == "yes")
    incorrect_judgments = sum(1 for r in results if r.get("correct") == "no")

    return {
        "total_questions": total,
        "judge_accuracy": judge_accuracy,
        "judge_correct": judge_correct,
        "average_confidence": avg_confidence,
        "average_rounds": avg_rounds,
        "termination_distribution": termination_counts,
        "judgment_distribution": {
            "yes": correct_judgments,
            "no": incorrect_judgments,
        },
    }


def save_hle_results(results: List[Dict], metrics: Dict, args):
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


def print_hle_summary(metrics: Dict[str, Any]):
    """Print HLE evaluation summary."""

    print("\n" + "=" * 60)
    print("📊 HLE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions: {metrics.get('total_questions', 0)}")
    print(f"Judge Accuracy: {metrics.get('judge_accuracy', 0):.2%}")
    print(f"Average Confidence: {metrics.get('average_confidence', 0):.1f}%")
    print(f"Average Rounds: {metrics.get('average_rounds', 0):.1f}")
    print(f"Evaluation Time: {metrics.get('evaluation_time', 0):.1f}s")

    print("\nTermination Reasons:")
    term_dist = metrics.get("termination_distribution", {})
    for reason, count in term_dist.items():
        print(f"  {reason}: {count}")

    print("\nJudgment Distribution:")
    judgment_dist = metrics.get("judgment_distribution", {})
    for judgment, count in judgment_dist.items():
        print(f"  {judgment}: {count}")

    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="Run HLE evaluation with DeepResearch + rLLM"
    )

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
    parser.add_argument(
        "--model", default=None, help="Model name for evaluation (default: gpt-4o)"
    )
    parser.add_argument(
        "--judge-model",
        default="o3-mini",
        help="Model name for judge (default: o3-mini, aligned with Tongyi)",
    )
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument(
        "--api-key", default=None, help="API key (uses env vars if not provided)"
    )

    # Execution options
    parser.add_argument(
        "--parallel-tasks", type=int, default=4, help="Number of parallel tasks"
    )
    parser.add_argument(
        "--output-dir", default="./hle_outputs", help="Output directory for results"
    )

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
