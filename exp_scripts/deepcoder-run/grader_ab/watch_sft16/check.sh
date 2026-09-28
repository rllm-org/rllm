#!/usr/bin/env bash
# Read-only snapshot of the Eval->Curate->SFT smoke on .16. Never touches processes.
ssh -o ConnectTimeout=15 10.225.68.16 '
P=$(pgrep -f "^bash .*run_(eval|pipeline|curate|sft)\.sh" | head -1)
R=/home/yanan/.deepcoder-sft-pipeline; [ -n "$P" ] && R=$(tr "\0" "\n" < /proc/$P/environ | sed -n "s/^WORK_ROOT=//p" | head -1); R=${R:-/home/yanan/.deepcoder-sft-pipeline}
C=/home/yanan/.deepcoder-checkpoints
echo "T=$(date +%T) WORK_ROOT=$R"
A=$(ps -eo args | grep -vE "bash -c|grep|check.sh")
echo "PROCS: vllm=$(grep -cE "vllm serve" <<<"$A") eval=$(grep -cE "eval_rollouts.py" <<<"$A") curate=$(grep -cE "curate_dataset.py" <<<"$A") sft=$(grep -cE "rllm.cli.main sft" <<<"$A") drivers=$(grep -cE "^bash .*run_(eval|pipeline|curate|sft)\.sh" <<<"$A")"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr "\n" " ")"
echo "RAM: $(free -g | awk "/Mem:/{print \$3\"/\"\$2\" GB\"}")"
if [ -d $R ]; then
  echo "WORK: $(ls $R 2>/dev/null | tr "\n" " ")"
  for d in $R/eval_runs/*/; do [ -d "$d" ] && echo "RUN $(basename $d): episodes=$(ls $d/episodes 2>/dev/null | wc -l) results=$([ -f $d/results.json ] && echo yes || echo no) meta=$([ -f $d/meta.json ] && echo yes || echo no)"; done
  for f in $R/*.eval.log $R/*.server.log; do [ -f "$f" ] && echo "LOG $(basename $f): $(stat -c%s $f)B last=$(tail -c 300 "$f" | tr "\n" " " | cut -c1-200)"; done
  for f in $R/*.eval.log $R/*.server.log; do [ -f "$f" ] && { n=$(grep -acE "Traceback|RuntimeError|refusing|OutOfMemory|CUDA out of memory|Error" "$f"); [ "$n" -gt 0 ] && echo "ERRORS in $(basename $f): $n -> $(grep -aE "Traceback|RuntimeError|refusing|OutOfMemory|CUDA out of memory" "$f" | tail -2 | cut -c1-200 | tr "\n" " | ")"; }; done
  for d in $R/eval_runs/*-sft-data/; do [ -d "$d" ] && echo "CURATED $(basename $d): $(ls $d | tr "\n" " ")"; done
else echo "WORK: (not created yet)"; fi
for d in $C/*sft*/; do [ -d "$d" ] && echo "SFT_CKPT $(basename $d): $(ls $d | tr "\n" " ")"; done
' 2>&1
