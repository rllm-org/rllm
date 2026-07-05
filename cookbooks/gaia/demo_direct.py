"""Direct-provider GAIA demo — bypasses the rLLM eval gateway.

Why: `rllm eval` routes model calls through a LiteLLM proxy configured with
`drop_params: True` (rllm/eval/proxy.py), which drops the `tools` parameter and
does not surface `tool_calls` back to the agent — so a tool-using agent can't
function-call through `rllm eval` today. This script runs the SAME agent loop
(`run_tool_loop`) and the SAME scorer (`question_scorer`) against the provider
directly, to demonstrate search -> browse -> answer end to end.

Env:
    OPENAI_API_KEY   provider key (works for any OpenAI-compatible endpoint)
    OPENAI_BASE_URL  optional (e.g. https://integrate.api.nvidia.com/v1)
    MODEL            model id (default: gpt-4o-mini)
    TAVILY_API_KEY   web search + extract
    HF_TOKEN         GAIA is gated

Run:
    python cookbooks/gaia/demo_direct.py --n 3
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from openai import OpenAI

from cookbooks.gaia.agent import run_tool_loop
from rllm.eval.reward_fns.gaia import _strip_answer_prefix, question_scorer
from rllm.tools.web_tools.tavily_tool import TavilyExtractTool, TavilySearchTool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of validation questions")
    args = ap.parse_args()

    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None, api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("MODEL", "gpt-4o-mini")
    tools = [TavilySearchTool(), TavilyExtractTool()]

    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=os.getenv("HF_TOKEN"))
    rows = [r for r in ds if not (r.get("file_name") or "").strip()][: args.n]  # text-only subset

    correct = 0
    for i, row in enumerate(rows, 1):
        question, gt = row["Question"], row["Final answer"]
        steps, answer = run_tool_loop(client, model, tools, question)
        pred = _strip_answer_prefix(answer)
        ok = question_scorer(pred, gt)
        correct += ok
        tool_steps = [s for s in steps if not s.done]
        print(f"\n[{i}] tool_calls={len(tool_steps)}  correct={ok}")
        print(f"    Q : {question[:110]}...")
        print(f"    -> pred={pred!r}   gt={gt!r}")

    print(f"\nAccuracy: {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
