#!/usr/bin/env bash
# Read-only: pulls the .24-local log over ssh and summarises grader timing. Never touches the trainer.
LOG=/home/yanan/.deepcoder-checkpoints/deepcoder-grpo-b16-n8-24-save64.log
ssh -o ConnectTimeout=15 10.225.68.24 "cat $LOG" 2>/dev/null | /home/yanan/agents/rllm/.venv/bin/python -c '
import re,sys,statistics as st
pat=re.compile(r"Rollout completed\. Rewards: \[deepcoder: ([0-9.]+)\] in (\d+)s .*?evaluator=(\d+)s")
step=0;a=0;recs=[];tb=0
for ln in sys.stdin:
    m=re.search(r"step:(\d+) - ",ln)
    if m: step=int(m.group(1))
    if "timeout occured: alarm went off" in ln: a+=1
    if "Traceback" in ln: tb+=1
    m=pat.search(ln)
    if m: recs.append((step,float(m.group(1)),int(m.group(3))))
n=len(recs); ev=sorted(r[2] for r in recs)
q=lambda f: ev[min(n-1,int(n*f))] if n else -1
slow=sum(1 for e in ev if e>=12)
print(f"step={step} rollouts={n} alarms={a} alarm_rate={(a/n*100 if n else 0):.2f}% eval_ge12={slow} ({(slow/n*100 if n else 0):.2f}%) p90={q(.9)}s p95={q(.95)}s max={(ev[-1] if n else -1)}s reward={(st.mean(r[1] for r in recs) if n else 0):.4f} tracebacks={tb}")
'
