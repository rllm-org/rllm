"""Fail-fast preflight for the FinQA judge.

finqa_eval._call_judge swallows every exception and returns 0.0/False, so a
misconfigured judge shows up as a plausible-looking 0% score instead of an
error. This exercises both judge paths against known-good and known-bad
inputs and crashes with expected/actual/why/fix if the judge is not live.

Run:  python preflight_judge.py       (after sourcing finqa_env.sh)
"""
import os, sys, time

def die(what, expected, actual, why, fix):
    print(f"\nPREFLIGHT FAILED: {what}", file=sys.stderr)
    print(f"  expected: {expected}", file=sys.stderr)
    print(f"  actual  : {actual}", file=sys.stderr)
    print(f"  why     : {why}", file=sys.stderr)
    print(f"  fix     : {fix}", file=sys.stderr)
    sys.exit(1)

for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "FINQA_JUDGE_MODEL", "FINQA_MULTI_TABLE_JUDGE_MODEL"):
    if not os.environ.get(var):
        die(f"env {var} unset", "non-empty", "unset",
            "finqa_eval builds its judge client at import time from env only",
            "source finqa_env.sh")

import finqa_eval as E

if E._JUDGE_CLIENT is None:
    die("judge client is None", "an openai.OpenAI instance", "None",
        "OPENAI_API_KEY was missing when finqa_eval was imported",
        "export OPENAI_API_KEY before importing")

print(f"judge base_url = {E._JUDGE_CLIENT.base_url}")
print(f"single-table model = {E.JUDGE_MODEL}")
print(f"multi-table  model = {E.MULTI_TABLE_JUDGE_MODEL}")

Q = "What was total revenue in fiscal 2023?"
GOOD = "391,035 million USD"
BAD  = "12 million USD"
TRUTH = "391,035 million USD"

def single(ans):
    return E._call_judge(E.CORRECTNESS_PROMPT,
        f"question : {Q}\nmodel response : {ans}\nlabel : {TRUTH}", multi_table=False)

print("\n[1/3] single-table, CORRECT answer ...", end=" ", flush=True)
t=time.time(); v,_ = single(GOOD); print(f"verdict={v} ({time.time()-t:.1f}s)")
if v is not True:
    die("single-table judge rejected a correct answer", "True", repr(v),
        "either the judge endpoint/model is wrong, or _call_judge raised and was swallowed",
        "check FINQA_JUDGE_MODEL matches a model served at OPENAI_BASE_URL")

print("[2/3] single-table, WRONG answer   ...", end=" ", flush=True)
t=time.time(); v,_ = single(BAD); print(f"verdict={v} ({time.time()-t:.1f}s)")
if v is not False:
    die("single-table judge accepted a wrong answer", "False", repr(v),
        "judge is not discriminating - reward signal would be constant",
        "inspect prompts/correctness prompt and the judge model's output")

print("[3/3] multi-table rubric           ...", end=" ", flush=True)
t=time.time()
score, rubric = E._call_judge(E.MULTI_TABLE_CORRECTNESS_PROMPT,
    f"question : {Q}\nmodel response : {GOOD}\nlabel : {TRUTH}", multi_table=True)
print(f"score={score:.3f} ({time.time()-t:.1f}s)")
if not rubric:
    die("multi-table judge returned an empty rubric", "a dict with the 6 rubric keys", "{}",
        "structured output (json_schema) failed or the response was not valid JSON",
        "verify the endpoint supports text.format=json_schema on /v1/responses")
missing = [k for k in E.CORRECTNESS_WEIGHTS if k not in rubric]
if missing:
    die("multi-table rubric missing keys", "all 6 rubric keys", f"missing {missing}",
        "strict json_schema was not enforced by the server",
        "check vLLM guided-decoding support for this model")
if score <= 0.0:
    die("multi-table judge scored a correct answer 0.0", "> 0.0", f"{score}",
        "scores parsed but all zero - likely a scale mismatch (code divides by 100)",
        "inspect the raw rubric values")

print("\nPREFLIGHT OK - judge is live and discriminating.")
