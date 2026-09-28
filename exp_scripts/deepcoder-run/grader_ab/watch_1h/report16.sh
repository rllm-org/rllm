#!/usr/bin/env bash
D=/home/yanan/agents/rllm/exp_scripts/deepcoder-run; W=$D/grader_ab/watch_1h
echo "=== $(date -Is)  .16 PRPO-safe since launch 12:12 (watch since $(cat $W/start16_ts)) ==="
PID_NOW=$(ssh -o ConnectTimeout=10 10.225.68.16 'pgrep -f "[t]rain_compatible.py" | head -1'); echo "trainer pid: start=$(cat $W/prpo16_pid_start) now=${PID_NOW:-NONE} $( [ "$PID_NOW" = "$(cat $W/prpo16_pid_start)" ] && echo '(same process)' || echo '(CHANGED / DEAD)')"
/home/yanan/agents/rllm/.venv/bin/python - "$D/logs/prpo16-safe.log" <<'PY'
import re,sys
pat=re.compile(r"Rollout completed\. Rewards: \[deepcoder: ([0-9.]+)\] in (\d+)s .*?evaluator=(\d+)s")
ev=[];al=0;steps=set();tb=0;rw=[]
for ln in open(sys.argv[1],errors="replace"):
    if "timeout occured: alarm went off" in ln: al+=1
    if "Traceback" in ln: tb+=1
    m=re.search(r"step:(\d+) - ",ln)
    if m: steps.add(int(m.group(1)))
    m=pat.search(ln)
    if m: ev.append(int(m.group(3))); rw.append(float(m.group(1)))
n=len(ev); s=sorted(ev)
def q(f): return s[min(n-1,int(n*f))] if n else -1
print(f"  steps={len(steps)} {('(max '+str(max(steps))+')') if steps else ''}  rollouts={n}  reward mean={(sum(rw)/n if n else 0):.4f}  eval p50={q(.5)}s p90={q(.9)}s p95={q(.95)}s max={(s[-1] if n else -1)}s  eval>=12s={sum(1 for e in ev if e>=12)} ({(sum(1 for e in ev if e>=12)/n*100 if n else 0):.2f}%)  alarms={al} ({(al/n*100 if n else 0):.2f}%)  tracebacks={tb}")
PY
