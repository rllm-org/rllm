#!/usr/bin/env bash
ssh -o ConnectTimeout=15 10.225.68.24 'FULL=/home/yanan/.deepcoder-checkpoints/deepcoder-grpo-b16-n8-24-c64-drop-save64.log; P=$(cat /home/yanan/.deepcoder-checkpoints/deepcoder-grpo-b16-n8-24-c64-drop-save64.pid)
# The log is appended across resumes; analyse only the segment after the last resume offset.
OFF=$(cat ${FULL%.log}.offset 2>/dev/null || echo 0); L=/tmp/grpo24_current_segment.log; tail -c +$((OFF+1)) $FULL > $L
echo "T=$(date +%T) pid=$P alive=$(kill -0 $P 2>/dev/null && echo yes || echo NO) step=$(grep -aoE "training/global_step:[0-9]+" $L | tail -1 | cut -d: -f2)"
echo "RAM: $(free -g | awk "/Mem:/{print \$3\"/\"\$2\" GB\"}")  biggest_grader_child=$(ps -eo rss,cmd | grep "[f]orkserver import main" | sort -rn | head -1 | awk "{printf \"%.1f GB\", \$1/1048576}")"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr "\n" " ")"
echo "CFG: $(grep -aoE "compact_filtering.enable +. +(True|False)|n_parallel_tasks +. +[0-9]+" $L | head -2 | tr -s " " | tr "\n" ";")"
ok=$(grep -ac "Rollout completed" $L); al=$(grep -ac "alarm went off" $L); hit=$(grep -ac "grader-drop" $L); retry=$(grep -ac "Attempt [0-9]/[0-9] failed" $L); err=$(grep -ac "Termination: TerminationReason.ERROR" $L); tb=$(grep -ac Traceback $L)
calls=$((ok+hit))
python3 - "$ok" "$al" "$hit" "$retry" "$err" "$tb" <<'"'"'PY'"'"'
import sys
ok,al,hit,retry,err,tb=map(int,sys.argv[1:])
calls=ok+hit
pct=lambda a,b: f"{a/b*100:.2f}% ({a}/{b})" if b else "n/a"
print(f"GRADER: calls={calls} alarm_lines={al} timeout_hits(grader-drop)={pct(hit,calls)}")
print(f"ROLLOUTS: completed={ok} retried_after_timeout={retry} dropped_as_ERROR={pct(err,ok+err)} tracebacks={tb}")
PY
echo "STEP_DROP_FRACTION(batch/termination_reason/error, last 5 steps): $(grep -aoE "batch/termination_reason/error[^0-9]+[0-9.]+" $L | grep -oE "[0-9.]+$" | tail -5 | tr "\n" " ")"
echo "STEP_TIMES: $(grep -aoE "timing_s/step:(np\.float64\()?[0-9.]+" $L | grep -oE "[0-9.]+$" | tail -5 | cut -d. -f1 | tr "\n" " ")   REWARD: $(grep -aoE "critic/score/mean:[0-9.]+" $L | tail -3 | cut -d: -f2 | tr "\n" " ")"
C=${FULL%.log}; echo "CKPTS: $(ls -d $C/global_step_* 2>/dev/null | xargs -rn1 basename | sed s/global_step_// | sort -n | tr "\n" " ") latest=$(cat $C/latest_checkpointed_iteration.txt 2>/dev/null) files_latest=$(find $C/global_step_$(cat $C/latest_checkpointed_iteration.txt 2>/dev/null) -type f 2>/dev/null | wc -l)"
echo "LLM_CALL_FAILED(600s timeouts, scored 0): $(grep -ac "LLM call failed" $L)"
grep -aE "Error executing job|killed due to memory pressure|OutOfMemoryError|CUDA out of memory" $L | tail -2 | cut -c1-200 | sed "s/^/FATAL: /"'
