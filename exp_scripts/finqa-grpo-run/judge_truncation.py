"""Does hitting max_output_tokens actually change the judge's verdict?

The judge endpoint reports finished_reason="length" on 6.3% of requests, i.e.
the response was cut off at max_output_tokens=512. finqa_eval._call_judge parses
whatever text came back with

    ("true" in text) and ("false" not in text)

so a truncated deliberation could yield an arbitrary verdict. That is a
plausible story, not a measurement. This measures it.

Method: send the same case twice — once at the production cap (512) and once at
a cap high enough that truncation cannot occur (4096) — and compare. Cases are
built to resemble what actually causes long judge responses: verbose agent
answers carrying reasoning, several numbers, units and hedging, rather than the
bare ground-truth strings that finish in ~59 tokens.

Usage:
    python judge_truncation.py [n_rows]
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import openai
import pandas as pd

sys.path.insert(0, "/home/yanan/agents/rllm/cookbooks/finqa")
import finqa_eval as E  # noqa: E402

DATA = "/home/yanan/agents/rllm/cookbooks/finqa/data/train_finqa.csv"
MODEL = os.environ.get("FINQA_JUDGE_MODEL", "Qwen/Qwen3.8-27B-FP8")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("OPENAI_API_KEY", "yyy")

PROD_CAP = 512    # what finqa_eval._call_judge uses
HIGH_CAP = 4096   # high enough that truncation cannot occur
CONCURRENCY = 4


def verbose_answer(row) -> str:
    """Mimic a real agent answer: reasoning, several numbers, units, hedging."""
    a = str(row["answer"])
    return (
        f"Let me work through this for {row['company']}. I first listed the available "
        f"tables, then inspected the schema and found the relevant rows. The figures I "
        f"pulled were for both comparison periods, and I had to reconcile a couple of "
        f"line items that were reported in thousands versus millions. After adjusting "
        f"for that and recomputing the difference, and double-checking the sign "
        f"convention because the table reports outflows as positive, I arrive at the "
        f"result below. Note there is some ambiguity about whether the prior-period "
        f"figure should be restated, but using the as-reported values:\n\n"
        f"FINAL ANSWER: {a}"
    )


def ask(client, question, answer, label, cap):
    try:
        r = client.responses.create(
            model=MODEL,
            instructions=E.CORRECTNESS_PROMPT,
            input=f"question : {question}\nmodel response : {answer}\nlabel : {label}",
            max_output_tokens=cap,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        raw = getattr(r, "output_text", "") or ""
        status = getattr(r, "status", None)
        inc = getattr(r, "incomplete_details", None)
        reason = getattr(inc, "reason", None) if inc else None
        finish = reason or status or "unknown"
        t = raw.lower()
        verdict = ("true" in t) and ("false" not in t)  # exactly what _call_judge does
        return finish, verdict, len(raw)
    except Exception as e:
        return f"exception:{type(e).__name__}", None, 0


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    df = pd.read_csv(DATA)
    rows = [df.iloc[i * 173 % len(df)] for i in range(n)]

    client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY) if BASE_URL else openai.OpenAI()
    print(f"judge  : {MODEL}")
    print(f"caps   : prod={PROD_CAP}  high={HIGH_CAP}")
    print(f"cases  : {n} verbose answers (correct ground truth, so both should say True)\n")

    def one(r):
        q, lab = r["user_query"], str(r["answer"])
        ans = verbose_answer(r)
        return ask(client, q, ans, lab, PROD_CAP), ask(client, q, ans, lab, HIGH_CAP)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        res = list(ex.map(one, rows))

    fin_prod = defaultdict(int)
    fin_high = defaultdict(int)
    truncated, flipped, agree = [], 0, 0
    for (fp, vp, lp), (fh, vh, lh) in res:
        fin_prod[fp] += 1
        fin_high[fh] += 1
        if vp is None or vh is None:
            continue
        if vp == vh:
            agree += 1
        if fp == "max_output_tokens":
            truncated.append((vp, vh, lp, lh))
            if vp != vh:
                flipped += 1

    print("=== finish reason at each cap ===")
    print(f"  prod cap {PROD_CAP}: " + ", ".join(f"{k}={v}" for k, v in sorted(fin_prod.items(), key=lambda kv: -kv[1])))
    print(f"  high cap {HIGH_CAP}: " + ", ".join(f"{k}={v}" for k, v in sorted(fin_high.items(), key=lambda kv: -kv[1])))

    comparable = sum(1 for (_, vp, _), (_, vh, _) in res if vp is not None and vh is not None)
    print(f"\n=== verdict agreement between the two caps ===")
    print(f"  {agree}/{comparable} = {agree / max(1, comparable):.1%}")

    print(f"\n=== among cases that WERE truncated at {PROD_CAP} ===")
    if not truncated:
        print("  none truncated in this sample — cannot measure flip rate here")
    else:
        print(f"  truncated: {len(truncated)}")
        print(f"  verdict flipped vs untruncated: {flipped}/{len(truncated)} = {flipped / len(truncated):.1%}")
        for vp, vh, lp, lh in truncated[:8]:
            mark = "FLIP" if vp != vh else "same"
            print(f"    {mark}  prod={vp} ({lp} chars)  high={vh} ({lh} chars)")

    # ground truth here is always a correct answer, so True is the right verdict
    for label, idx in (("prod", 0), ("high", 1)):
        ok = sum(1 for r in res if r[idx][1] is True)
        tot = sum(1 for r in res if r[idx][1] is not None)
        print(f"\n  accuracy at {label} cap (all cases are genuinely correct): {ok}/{tot} = {ok / max(1, tot):.1%}")


if __name__ == "__main__":
    main()
