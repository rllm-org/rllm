"""Gaia dataset evaluator using StrandsAgent and RLLM integration.

This module provides evaluation capabilities for the Gaia dataset using
the existing Strands + RLLM integration.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from strands.handlers.callback_handler import null_callback_handler

from rllm.engine.rollout import OpenAIEngine
from rllm.integrations.strands import RLLMModel, StrandsAgent

# Add parent directory to path to import strands_workflow
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from strands_workflow import StrandsWorkflow


@dataclass
class GaiaSample:
    """Represents a single Gaia dataset sample."""

    task_id: str
    question: str
    answer: str
    file_name: str
    level: str | None = None
    file_path: str | None = None


@dataclass
class EvaluationResult:
    """Represents the evaluation result for a single sample."""

    task_id: str
    question: str
    ground_truth: str
    model_response: str
    is_correct: bool
    f1_score: float
    exact_match: bool
    tool_usage: dict[str, int]
    execution_time: float
    error_message: str | None = None


class GaiaEvaluator:
    """Evaluator for Gaia dataset using StrandsAgent."""

    def __init__(self, rollout_engine: OpenAIEngine, tools: list[Any], system_prompt: str | None = None, max_samples: int | None = None, output_dir: str = "outputs/gaia_eval"):
        self.rollout_engine = rollout_engine
        self.max_samples = max_samples
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        default_system_prompt = "You are a helpful agent solving web-based tasks. Use tools when beneficial to find information and solve problems. Provide clear, accurate answers based on the information you gather."

        self.base_workflow_args = {"agent_cls": StrandsAgent, "base_agent_args": {"tools": tools, "system_prompt": system_prompt or default_system_prompt, "callback_handler": null_callback_handler}}

        self.results: list[EvaluationResult] = []
        self.tool_usage_stats: dict[str, int] = {}
        self._rollout_engine = rollout_engine

    def load_gaia_dataset(self, dataset_path: str) -> list[GaiaSample]:
        """Load Gaia dataset from JSON file.

        Args:
            dataset_path: Path to the Gaia dataset JSON file

        Returns:
            List of GaiaSample objects
        """
        with open(dataset_path) as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = GaiaSample(task_id=item.get("task_id", ""), question=item.get("problem", item.get("Question", "")), answer=item.get("tests", item.get("Final answer", "")), file_name=item.get("file_name", ""), level=item.get("Level", None), file_path=item.get("file_path", None))
            samples.append(sample)

        if self.max_samples:
            samples = samples[: self.max_samples]

        print(f"Loaded {len(samples)} samples from {dataset_path}")
        return samples

    async def evaluate_single_sample(self, sample: GaiaSample) -> EvaluationResult:
        start_time = time.time()

        try:
            task_dict = {"task": sample.question}
            task_id = f"gaia_task_{sample.task_id}"

            # Create fresh workflow with fresh model for zero contamination
            fresh_model = RLLMModel(rollout_engine=self._rollout_engine, model_id=f"gaia-evaluator-{sample.task_id}")

            workflow_args = {"agent_cls": self.base_workflow_args["agent_cls"], "agent_args": {"model": fresh_model, **self.base_workflow_args["base_agent_args"]}}

            fresh_workflow = StrandsWorkflow(rollout_engine=self._rollout_engine, executor=ThreadPoolExecutor(max_workers=1), **workflow_args)

            episode = await fresh_workflow.run_with_termination_handling(task_dict, task_id)

            if not episode or not episode.trajectories:
                raise Exception("No episode or trajectory returned")

            trajectory = episode.trajectories[0][1]
            model_response = self._extract_response_from_trajectory(trajectory)
            is_correct, f1_score, exact_match = self._calculate_metrics(model_response, sample.answer)
            tool_usage = self._extract_tool_usage_from_trajectory(trajectory)

            return EvaluationResult(task_id=sample.task_id, question=sample.question, ground_truth=sample.answer, model_response=model_response, is_correct=is_correct, f1_score=f1_score, exact_match=exact_match, tool_usage=tool_usage, execution_time=time.time() - start_time)

        except Exception as e:
            return EvaluationResult(task_id=sample.task_id, question=sample.question, ground_truth=sample.answer, model_response="", is_correct=False, f1_score=0.0, exact_match=False, tool_usage={}, execution_time=time.time() - start_time, error_message=str(e))

    async def evaluate_dataset(self, dataset_path: str) -> list[EvaluationResult]:
        """Evaluate the entire Gaia dataset.

        Args:
            dataset_path: Path to the Gaia dataset JSON file

        Returns:
            List of EvaluationResult objects
        """
        samples = self.load_gaia_dataset(dataset_path)
        results = []

        print(f"Starting evaluation of {len(samples)} samples...")

        for i, sample in enumerate(samples):
            print(f"\n{'=' * 60}")
            print(f"📝 Sample {i + 1}/{len(samples)}: {sample.task_id}")
            print(f"❓ Question: {sample.question}")
            print(f"✅ Ground Truth: {sample.answer}")
            print()  # Clean separator

            result = await self.evaluate_single_sample(sample)
            results.append(result)

            # Print model response and evaluation result
            print(f"🤖 Model Response: {result.model_response}")
            print(f"📊 Result: {'✅ Correct' if result.is_correct else '❌ Incorrect'} | F1: {result.f1_score:.3f} | Exact Match: {'Yes' if result.exact_match else 'No'}")
            print(f"⏱️  Time: {result.execution_time:.2f}s", end="")
            if result.tool_usage:
                tools_summary = ", ".join([f"{tool}({count})" for tool, count in result.tool_usage.items()])
                print(f" | 🔧 Tools: {tools_summary}")
            else:
                print()
            if result.error_message:
                print(f"⚠️  Error: {result.error_message}")
            print(f"{'=' * 60}")

            # Update tool usage stats
            for tool_name, count in result.tool_usage.items():
                self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + count

        self.results = results
        return results

    def _extract_response_from_trajectory(self, trajectory) -> str:
        if not trajectory.steps:
            return ""

        final_response = ""
        for step in trajectory.steps:
            if step.model_response and step.model_response.strip():
                if not (step.model_response.startswith("[Tool:") or step.model_response.startswith("[Using tool:")):
                    final_response = step.model_response

            if step.action and isinstance(step.action, dict) and "final_response" in step.action:
                final_resp = step.action["final_response"]

                if isinstance(final_resp, dict) and "content" in final_resp:
                    content = final_resp["content"]
                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text = item["text"]
                                if not (text.startswith("[Tool:") or text.startswith("[Using tool:")):
                                    text_parts.append(text)
                        if text_parts:
                            final_response = " ".join(text_parts)
                elif isinstance(final_resp, str):
                    final_response = final_resp

        return final_response

    def _extract_tool_usage_from_trajectory(self, trajectory) -> dict[str, int]:
        tool_usage = {}
        for step in trajectory.steps:
            if hasattr(step, "action") and step.action and isinstance(step.action, dict):
                if step.action.get("type") == "tool_calls":
                    for tool_call in step.action.get("tool_calls", []):
                        tool_name = tool_call.get("name")
                        if tool_name:
                            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
                elif step.action.get("type") == "tool_call":
                    tool_name = step.action.get("tool_name")
                    if tool_name:
                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        return tool_usage

    def _calculate_metrics(self, model_response: str, ground_truth: str) -> tuple[bool, float, bool]:
        exact_match = model_response.strip().lower() == ground_truth.strip().lower()

        if exact_match:
            f1_score = 1.0
        else:
            model_words = set(model_response.lower().split())
            gt_words = set(ground_truth.lower().split())

            if not model_words or not gt_words:
                f1_score = 0.0
            else:
                intersection = len(model_words.intersection(gt_words))
                precision = intersection / len(model_words) if model_words else 0
                recall = intersection / len(gt_words) if gt_words else 0

                if precision + recall == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)

        is_correct = exact_match or f1_score > 0.5
        return is_correct, f1_score, exact_match

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.results:
            return {"error": "No results to report"}

        total_samples = len(self.results)
        correct_samples = sum(1 for r in self.results if r.is_correct)
        exact_matches = sum(1 for r in self.results if r.exact_match)

        # Calculate average metrics
        avg_f1 = sum(r.f1_score for r in self.results) / total_samples
        avg_execution_time = sum(r.execution_time for r in self.results) / total_samples

        # Error analysis
        errors = [r for r in self.results if r.error_message]

        report = {"summary": {"total_samples": total_samples, "correct_samples": correct_samples, "accuracy": correct_samples / total_samples, "exact_match_rate": exact_matches / total_samples, "average_f1_score": avg_f1, "average_execution_time": avg_execution_time}, "tool_usage": self.tool_usage_stats, "error_count": len(errors), "results": [asdict(r) for r in self.results]}

        return report

    def save_results(self, filename: str | None = None):
        """Save evaluation results to files."""
        if not filename:
            timestamp = int(time.time())
            filename = f"gaia_eval_results_{timestamp}"

        # Save detailed results as JSON
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, "w") as f:
            json.dump(self.generate_report(), f, indent=2)

        # Save summary as CSV
        csv_path = os.path.join(self.output_dir, f"{filename}_summary.csv")
        summary_data = []
        for result in self.results:
            summary_data.append({"task_id": result.task_id, "is_correct": result.is_correct, "f1_score": result.f1_score, "exact_match": result.exact_match, "execution_time": result.execution_time, "error_message": result.error_message or ""})

        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)

        print("Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")

        return json_path, csv_path
