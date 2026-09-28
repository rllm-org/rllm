#!/usr/bin/env bash
D=/home/yanan/agents/rllm/exp_scripts/deepcoder-run; W=$D/grader_ab/watch_1h
echo "=== $(date -Is)  window since $(cat $W/start_ts) ==="
PID_NOW=$(ssh -o ConnectTimeout=10 10.225.68.24 'pgrep -f "[t]rain_compatible.py" | head -1'); echo "prpo trainer pid: start=$(cat $W/prpo_pid_start) now=$PID_NOW $( [ "$PID_NOW" = "$(cat $W/prpo_pid_start)" ] && echo '(same process, no restart)' || echo '(RESTARTED)')"
for a in prpo grpo; do
  tail -n +"$(( $(cat $W/${a}_start_line) + 1 ))" $D/logs/$a.log > $W/${a}_window.log
  /home/yanan/agents/rllm/.venv/bin/python - "$a" "$W/${a}_window.log" <<'PY'
import re,sys
arm,f=sys.argv[1],sys.argv[2]
pat=re.compile(r"Rollout completed\. Rewards: \[deepcoder: ([0-9.]+)\] in (\d+)s .*?evaluator=(\d+)s")
ev=[];al=0;steps=set()
for ln in open(f,errors="replace"):
    if "timeout occured: alarm went off" in ln: al+=1
    m=re.search(r"step:(\d+) - ",ln)
    if m: steps.add(int(m.group(1)))
    m=pat.search(ln)
    if m: ev.append(int(m.group(3)))
n=len(ev); ev_s=sorted(ev)
def q(f): return ev_s[min(n-1,int(n*f))] if n else -1
print(f"  {arm}: steps completed in window={len(steps)} {('('+str(min(steps))+'-'+str(max(steps))+')') if steps else ''}  rollouts={n}  eval p90={q(.9)}s p95={q(.95)}s  eval>=12s={sum(1 for e in ev if e>=12)} ({(sum(1 for e in ev if e>=12)/n*100 if n else 0):.2f}%)  alarms={al} ({(al/n*100 if n else 0):.2f}%)")
PY
done
