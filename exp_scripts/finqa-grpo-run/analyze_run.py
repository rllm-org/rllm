"""Extract per-step timing + reward stats from a verl/rLLM training log.

Usage: python analyze_run.py smoke.log
"""
import re, sys, json

path = sys.argv[1] if len(sys.argv) > 1 else "smoke.log"
txt = open(path, errors="replace").read()

# verl prints one 'step:N - key:val - key:val ...' line per training step
step_lines = [l for l in txt.splitlines() if re.match(r"^\s*step:\d+\s+-", l)]
print(f"training steps logged: {len(step_lines)}\n")

WANT = ["timing_s/step", "timing_s/gen", "timing_s/update_actor", "timing_s/adv",
        "critic/rewards/mean", "critic/score/mean", "critic/advantages/mean",
        "critic/advantages/std", "response_length/mean", "prompt_length/mean",
        "actor/pg_loss", "actor/grad_norm", "perf/throughput"]

rows = []
for l in step_lines:
    d = dict(re.findall(r"([\w/@\.\-]+):([-\d\.eE\+naN]+)", l))
    rows.append(d)

if rows:
    keys = [k for k in WANT if any(k in r for r in rows)]
    extra = sorted({k for r in rows for k in r} - set(keys) - {"step"})
    hdr = "| step | " + " | ".join(k.split("/")[-1][:14] for k in keys) + " |"
    print(hdr); print("|" + "-"*(len(hdr)-2) + "|")
    for r in rows:
        vals = " | ".join(f"{float(r[k]):.4g}" if k in r else "-" for k in keys)
        print(f"| {r.get('step','?'):>4} | {vals} |")
    print(f"\nfull key list ({len(extra)} more): {extra[:40]}")

# reward distribution sanity: constant reward == broken signal
rew = [float(r[k]) for r in rows for k in ("critic/rewards/mean","critic/score/mean") if k in r]
if rew:
    print(f"\nreward mean across steps: min={min(rew):.4f} max={max(rew):.4f}")
    if len(set(rew)) == 1:
        print("  !! WARNING: reward identical across every step - check the judge")

for pat, label in [(r"EnrichMismatchError", "EnrichMismatchError (missing token_ids)"),
                   (r"CUDA out of memory|OutOfMemoryError", "CUDA OOM"),
                   (r"Traceback \(most recent call last\)", "tracebacks"),
                   (r"No traces found", "episodes with no gateway traces")]:
    n = len(re.findall(pat, txt))
    if n:
        print(f"  {label}: {n} occurrence(s)")
