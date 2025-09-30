"""
HLE Results Viewer - Display evaluation results in a clean, readable format

This script loads HLE evaluation results and displays them in a concise format,
showing only the most important information without the verbose details.
"""

import json
import sys
import argparse
from typing import Dict, Any


def load_results(results_file: str) -> Dict[str, Any]:
    """Load HLE results from JSON file."""
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        sys.exit(1)


def print_summary(data: Dict[str, Any]):
    """Print evaluation summary."""
    metadata = data.get("metadata", {})
    metrics = data.get("metrics", {})

    print("🎯 HLE EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Dataset: {metadata.get('dataset', 'Unknown')}")
    print(f"Model: {metadata.get('model', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"Total Questions: {metadata.get('total_questions', 0)}")
    print()

    print("📊 Performance Metrics:")
    print(f"Judge Accuracy: {metrics.get('judge_accuracy', 0):.2%}")
    print(f"Average Rating: {metrics.get('average_rating', 0):.2f}/5.0")
    print(f"Average Rounds: {metrics.get('average_rounds', 0):.1f}")
    print(f"Evaluation Time: {metrics.get('evaluation_time', 0):.1f}s")
    print()

    # Rating distribution
    print("📈 Rating Distribution:")
    rating_dist = metrics.get("rating_distribution", {})
    for rating in ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5"]:
        count = rating_dist.get(rating, 0)
        stars = "★" * count if count > 0 else ""
        print(f"  {rating.replace('rating_', '')} stars: {count:2d} {stars}")
    print()

    # Termination reasons
    print("🏁 Termination Reasons:")
    term_dist = metrics.get("termination_distribution", {})
    for reason, count in term_dist.items():
        print(f"  {reason}: {count}")
    print()


def print_detailed_results(data: Dict[str, Any], max_show: int = 5):
    """Print detailed results for individual questions."""
    results = data.get("results", [])

    print(f"📝 DETAILED RESULTS (showing first {min(max_show, len(results))})")
    print("=" * 50)

    for i, result in enumerate(results[:max_show]):
        print(f"\n🔍 Question {i + 1}:")
        print(f"Subject: {result.get('subject', 'Unknown')}")
        print(
            f"Rating: {result.get('rating', 0)}/5 {'✅' if result.get('is_correct', False) else '❌'}"
        )
        print(f"Rounds: {result.get('rounds', 0)}")
        print(f"Termination: {result.get('termination_reason', 'Unknown')}")

        # Truncate long texts
        question = result.get("question", "")[:150]
        if len(result.get("question", "")) > 150:
            question += "..."

        prediction = result.get("prediction", "")[:200]
        if len(result.get("prediction", "")) > 200:
            prediction += "..."

        reference = result.get("reference_answer", "")[:150]
        if len(result.get("reference_answer", "")) > 150:
            reference += "..."

        print(f"Q: {question}")
        print(f"A: {prediction}")
        print(f"Expected: {reference}")

        # Show judge reasoning (truncated)
        judgment = result.get("judgment", "")
        if judgment and len(judgment) > 300:
            # Extract key parts of judgment
            lines = judgment.split("\n")
            key_lines = [
                line
                for line in lines
                if "correct" in line.lower()
                or "accurate" in line.lower()
                or "rating" in line.lower()
            ][:2]
            if key_lines:
                print(f"Judge: {' '.join(key_lines)[:200]}...")
        elif judgment:
            print(f"Judge: {judgment[:200]}...")

        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="View HLE evaluation results")
    parser.add_argument("results_file", help="Path to HLE results JSON file")
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed results for individual questions",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=5,
        help="Maximum number of detailed results to show (default: 5)",
    )

    args = parser.parse_args()

    # Load results
    data = load_results(args.results_file)

    # Print summary
    print_summary(data)

    # Print detailed results if requested
    if args.detailed:
        print_detailed_results(data, args.max_show)
    else:
        print("💡 Use --detailed to see individual question results")


if __name__ == "__main__":
    main()
