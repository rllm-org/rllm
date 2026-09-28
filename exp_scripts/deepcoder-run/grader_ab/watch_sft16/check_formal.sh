#!/usr/bin/env bash
ssh -o ConnectTimeout=15 10.225.68.16 'R=/home/yanan/.deepcoder-sft-pipeline-formal
echo "T=$(date +%T) drivers=$(ps -eo cmd | grep -c "^bash .*run_formal_dual.sh") evals=$(ps -eo cmd | grep -c "[e]val_rollouts.py") vllm=$(ps -eo cmd | grep -c "[v]llm serve")"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr "\n" " ")"
for s in 0 1; do d=$R/eval_runs/base-train-full-k8-shard$s; if [ -f $d/progress.json ]; then echo "SHARD$s: $(tr -d "\n " < $d/progress.json) mtime=$(date -r $d/progress.json +%T)"; else echo "SHARD$s: no progress.json yet, episodes_on_disk=$(ls $d/episodes 2>/dev/null | wc -l)"; fi; done
for s in 0 1; do f=$R/base-train-full-k8-shard$s.eval.log; n=$(grep -acE "Traceback|RuntimeError|OutOfMemory|CUDA out of memory|refusing" $f); [ "$n" -gt 0 ] && echo "ERRORS shard$s: $n $(grep -aE "Traceback|RuntimeError|OutOfMemory|refusing" $f | tail -1 | cut -c1-160)"; done
echo "DISK: $(df -h /home/yanan | awk "NR==2{print \$4\" free\"}")  RUNSIZE: $(du -sh $R 2>/dev/null | cut -f1)"
echo "NOTE: alarms_total=$(cat $R/*.eval.log | grep -ac "alarm went off") rollout_errors=$(cat $R/*.eval.log | grep -ac "TerminationReason.ERROR")"'
