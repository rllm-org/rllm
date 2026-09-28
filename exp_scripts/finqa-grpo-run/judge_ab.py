"""A/B two LLM judges on the FinQA correctness rubric.

Runs entirely against external endpoints (Azure + .29:1702) — no GPU on .16, so
it cannot disturb a running training job.

Both judges agree trivially on clearly-right and clearly-wrong answers, so those
tell us nothing. The disagreement surface is *format tolerance* and *near
misses*, which is exactly what decides whether a judge inflates or deflates the
reward signal. Cases are therefore generated as controlled perturbations of real
ground-truth answers:

    exact         verbatim ground truth                 -> expect PASS
    reformatted   same value, $ / commas / units added  -> expect PASS (format tolerance)
    rounded       same value, one fewer decimal         -> expect PASS (tolerance)
    sign_flip     negated                               -> expect FAIL
    magnitude     off by 10x                            -> expect FAIL
    unrelated     a different number entirely           -> expect FAIL
    vague         hedge text, no number                 -> expect FAIL

Usage:
    python judge_ab.py [n_rows_per_category]
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

import openai
import pandas as pd

sys.path.insert(0, "/home/yanan/agents/rllm/cookbooks/finqa")
import finqa_eval as E  # noqa: E402

DATA = "/home/yanan/agents/rllm/cookbooks/finqa/data/train_finqa.csv"

JUDGES = {
    # name           (model,                  base_url,                      api_key)
    "qwen3.8-27b":  ("Qwen/Qwen3.8-27B-FP8", "http://10.225.68.29:1702/v1", "yyy"),
    "gpt-5.4-mini": ("gpt-5.4-mini", None, None),   # None -> Azure via env
    "gpt-5.6-luna": ("gpt-5.6-luna", None, None),
    # finqa.md's own single-table judge was gpt-5-nano, so this tier is the
    # closest analogue to the paper's configuration.
    "gpt-5.4-nano": ("gpt-5.4-nano", None, None),
}

# Modest concurrency: the qwen judge also serves the live training run, whose
# own judge calls are bursty (256 at the end of every step).
CONCURRENCY = 4

_NUM = re.compile(r"-?[\d,]*\.?\d+")


def perturb(ans: str) -> dict[str, str | None]:
    """Build one variant of `ans` per category. None = not applicable to this row."""
    a = str(ans).strip()
    m = _NUM.search(a)
    out: dict[str, str | None] = {"exact": a, "vague": "It depends on which period you look at; the figure moved somewhat."}
    if not m:
        return out
    raw = m.group(0)
    try:
        val = float(raw.replace(",", ""))
    except ValueError:
        return out
    sub = lambda new: a[: m.start()] + new + a[m.end() :]  # noqa: E731
    out["reformatted"] = sub(f"${abs(val):,.2f}" if val >= 0 else f"-${abs(val):,.2f}")
    out["rounded"] = sub(f"{round(val, 1):g}") if val != round(val, 1) else None
    out["sign_flip"] = sub(f"{-val:g}") if val != 0 else None
    out["magnitude"] = sub(f"{val * 10:g}") if val != 0 else None
    out["unrelated"] = sub(f"{val + 4173.5:g}")
    return out


EXPECT_PASS = {"exact", "reformatted", "rounded"}


def make_client(base_url, key):
    if base_url is None:
        return openai.OpenAI()  # Azure, from env
    return openai.OpenAI(base_url=base_url, api_key=key)


def ask(client, model, question, answer, label):
    """Returns (verdict, raw_text, meta). verdict is None when the response is
    not a clean completion (content filter, truncation, error)."""
    try:
        r = client.responses.create(
            model=model,
            instructions=E.CORRECTNESS_PROMPT,
            input=f"question : {question}\nmodel response : {answer}\nlabel : {label}",
            max_output_tokens=512,
            reasoning={"effort": "low"},
            text={"verbosity": "low"},
        )
        raw = getattr(r, "output_text", "") or ""

        # Record the termination state of EVERY response. In the Responses API
        # the chat-completions `finish_reason` is split across:
        #   r.status                      "completed" | "incomplete"
        #   r.incomplete_details.reason   "max_output_tokens" | "content_filter" | ...
        #   item.status per output item
        # A truncated or filtered response still yields text, and
        # finqa_eval._call_judge would happily parse it into a verdict --
        # on the .29:1702 judge, 6.2% of all requests already finish with
        # reason="length". So the state is captured, not inferred.
        status = getattr(r, "status", None)
        inc = getattr(r, "incomplete_details", None)
        inc_reason = getattr(inc, "reason", None) if inc else None
        item_status = [getattr(it, "status", None) for it in (getattr(r, "output", None) or [])]
        fin = inc_reason or status or "unknown"

        meta = {
            "status": status,
            "incomplete_reason": inc_reason,
            "item_status": item_status,
            "rawlen": len(raw),
            "finish": fin,
        }
        # Anything that is not a clean, non-empty completion is NOT a verdict.
        if status != "completed" or inc_reason or not raw.strip():
            return None, f"NOJUDGE finish={fin} status={status} rawlen={len(raw)}", meta
        t = raw.lower()
        return ("true" in t) and ("false" not in t), raw, meta
    except Exception as e:  # a judge that errors is itself a finding
        return None, f"ERROR {type(e).__name__}: {str(e)[:120]}", {"finish": "exception:" + type(e).__name__}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    df = pd.read_csv(DATA)
    rows = df.iloc[[i * 331 % len(df) for i in range(n)]]

    clients = {name: (make_client(url, key), model) for name, (model, url, key) in JUDGES.items()}
    names = list(clients)

    cases = []
    for _, r in rows.iterrows():
        for cat, ans in perturb(r["answer"]).items():
            if ans is None:
                continue
            cases.append((cat, r["user_query"], ans, str(r["answer"])))

    print(f"judges : {names}")
    print(f"cases  : {len(cases)} ({n} rows x categories)\n")

    from concurrent.futures import ThreadPoolExecutor

    def one(idx_case):
        i, (cat, q, ans, label) = idx_case
        res = {}
        for name in names:
            client, model = clients[name]
            res[name] = ask(client, model, q, ans, label)
        return i, cat, res

    results = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for i, cat, res in ex.map(one, enumerate(cases)):
            results[i] = (cat, res)
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(cases)}", flush=True)

    verdicts: dict[str, list] = {k: [] for k in names}
    raws: dict[str, list] = {k: [] for k in names}
    metas: dict[str, list] = {k: [] for k in names}
    cats = []
    for cat, res in results:
        cats.append(cat)
        for name in names:
            v, raw, meta = res[name]
            verdicts[name].append(v)
            raws[name].append(raw)
            metas[name].append(meta)

    idx = [i for i in range(len(cats)) if all(verdicts[n][i] is not None for n in names)]
    print(f"\n=== comparable cases: {len(idx)} / {len(cats)} ===")

    print("\n=== finish state of EVERY response ===")
    print(f"  {'judge':14s} " + "  ".join(f"{k:>18s}" for k in ["completed", "max_output_tokens", "content_filter", "other"]))
    for n in names:
        c = defaultdict(int)
        for m in metas[n]:
            f = str(m.get("finish"))
            key = f if f in ("completed", "max_output_tokens", "content_filter") else "other"
            c[key] += 1
        print(f"  {n:14s} " + "  ".join(f"{c[k]:>18d}" for k in ["completed", "max_output_tokens", "content_filter", "other"]))
    print("\n  full finish-reason breakdown:")
    for n in names:
        c = defaultdict(int)
        for m in metas[n]:
            c[str(m.get("finish"))] += 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
        print(f"    {n:14s} {detail}")

    print("\n=== refusals / non-verdicts (content filter, truncation, errors) ===")
    any_bad = False
    for n in names:
        bad = [(cats[i], raws[n][i]) for i in range(len(cats)) if verdicts[n][i] is None]
        print(f"  {n:14s} {len(bad):4d} / {len(cats)}")
        if bad:
            any_bad = True
            kinds = defaultdict(int)
            for c, r in bad:
                kinds[r.split(" rawlen")[0][:70]] += 1
            for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:4]:
                print(f"      x{v:<4d} {k}")
    if not any_bad:
        print("  none - every judge returned a usable verdict on every case")

    print("\n=== pairwise agreement ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ag = sum(1 for k in idx if verdicts[a][k] == verdicts[b][k])
            print(f"  {a:14s} vs {b:14s}  {ag}/{len(idx)} = {ag / max(1,len(idx)):.1%}")

    print("\n=== unanimity ===")
    unan = sum(1 for k in idx if len({verdicts[n][k] for n in names}) == 1)
    print(f"  all {len(names)} agree: {unan}/{len(idx)} = {unan / max(1,len(idx)):.1%}")

    print("\n=== by category (pass counts) ===")
    hdr = f"  {'category':12s} {'n':>4s} {'unan':>6s}" + "".join(f" {n:>14s}" for n in names) + "  expected"
    print(hdr)
    per = defaultdict(list)
    for k in idx:
        per[cats[k]].append(k)
    for c in ["exact", "reformatted", "rounded", "sign_flip", "magnitude", "unrelated", "vague"]:
        if c not in per:
            continue
        ks = per[c]
        u = sum(1 for k in ks if len({verdicts[n][k] for n in names}) == 1)
        row = f"  {c:12s} {len(ks):4d} {u/len(ks):6.0%}"
        for n in names:
            row += f" {sum(1 for k in ks if verdicts[n][k]):>8d} pass"
        row += f"  {'PASS' if c in EXPECT_PASS else 'FAIL':>8s}"
        print(row)

    print("\n=== strictness (pass rate) ===")
    for n in names:
        vv = [verdicts[n][k] for k in idx]
        print(f"  {n:14s} {sum(vv)}/{len(vv)} = {sum(vv)/max(1,len(vv)):.1%} pass")

    print("\n=== accuracy vs the expected label ===")
    for n in names:
        ok = sum(1 for k in idx if verdicts[n][k] == (cats[k] in EXPECT_PASS))
        print(f"  {n:14s} {ok}/{len(idx)} = {ok/max(1,len(idx)):.1%} correct")

    print("\n=== disagreement breakdown ===")
    d = defaultdict(int)
    for k in idx:
        if len({verdicts[n][k] for n in names}) > 1:
            d[(cats[k], tuple(verdicts[n][k] for n in names))] += 1
    if not d:
        print("  none")
    for (c, pat), k in sorted(d.items(), key=lambda kv: -kv[1]):
        print(f"  {c:12s} x{k:3d}   " + "  ".join(f"{n}={p}" for n, p in zip(names, pat)))

    # a verbose judge can flip a correct verdict just by using the word "false"
    print("\n=== 'false' appearing inside an otherwise-correct explanation ===")
    for n in names:
        bad = sum(1 for k in idx
                  if cats[k] in EXPECT_PASS and verdicts[n][k] is False
                  and "true" in raws[n][k].lower())
        print(f"  {n:14s} {bad} case(s) where the raw text contains both 'true' and 'false'")


if __name__ == "__main__":
    main()
