#!/usr/bin/env bash
# Three FinQA runs on .24, strictly sequential, each val(522)+test(558) one per GPU.
#
#   A  Qwen3-4B-Instruct-2507      MAX_TURNS=20  (matches the Qwen3.5 thinking-off run
#                                                 already measured; 2507 hit the cap on
#                                                 0.1% of tasks so the budget is immaterial)
#   B  Qwen3.5-4B thinking OFF     MAX_TURNS=50  RLLM_DISABLE_THINKING=1
#   C  Qwen3.5-4B thinking ON      MAX_TURNS=50  RLLM_DISABLE_THINKING=0
#
# Everything else is held fixed: temperature 0, top_p 1.0, seed 1234, judge
# gpt-5.4-nano, concurrency 32, max_model_len 12288, same splits, same flow.
# A run that scores exactly 0 is flagged loudly -- that is the silent-failure
# signature (empty content), not a capability result.
set -u
RD=/home/yanan/agents/rllm/exp_scripts/finqa-grpo-run
M2507=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
M35=/home/yanan/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
CHAIN_LOG="$RD/eval/chain-20260922.log"
mkdir -p "$RD/eval"
export VLLM_USE_FLASHINFER_SAMPLER=0

run_one() {  # $1=tag $2=model $3=max_turns $4=disable_thinking
  local tag="$1" model="$2" turns="$3" nothink="$4"
  echo "=== [$tag] START $(date '+%F %T %Z')  MAX_TURNS=$turns RLLM_DISABLE_THINKING=$nothink" | tee -a "$CHAIN_LOG"
  if [ -d "$RD/eval/$tag" ]; then mv "$RD/eval/$tag" "$RD/eval/$tag.superseded.$(date +%s)"; fi
  mkdir -p "$RD/eval/$tag"
  ( cd "$RD"
    export FINQA_MAX_TURNS="$turns"
    export RLLM_DISABLE_THINKING="$nothink"
    ./eval_parallel.sh "$model" "$tag" > "$RD/eval/$tag/launcher.log" 2>&1 )
  local rc=$?
  for s in val test; do
    local f="$RD/eval/$tag/$s.json"
    if [ -f "$f" ]; then
      python3 -c "
import json;d=json.load(open('$f'))
c,t=d['correct'],d['total']
flag=''
if c==0: flag='   <<< ZERO SCORE - suspect silent failure, inspect before trusting'
print(f'[$tag] $s: {100*c/t:.2f}% ({c}/{t}) errors={d[\"errors\"]}{flag}')" | tee -a "$CHAIN_LOG"
    else
      echo "[$tag] $s: MISSING result.json (rc=$rc)" | tee -a "$CHAIN_LOG"
    fi
  done
  echo "=== [$tag] END $(date '+%F %T %Z') rc=$rc" | tee -a "$CHAIN_LOG"
  # All three runs share one code path; a failure here is almost certainly
  # systemic, so stop rather than burn hours on two more broken runs.
  if [ ! -f "$RD/eval/$tag/val.json" ] || [ ! -f "$RD/eval/$tag/test.json" ]; then
    echo "=== CHAIN ABORTED: [$tag] produced no result; see $RD/eval/$tag/eval_*.log" | tee -a "$CHAIN_LOG"
    exit 1
  fi
}

run_one "2507-fair-t20"        "$M2507" 20 0
run_one "qwen35-nothink-t50"   "$M35"   50 1
run_one "qwen35-think-t50"     "$M35"   50 0
echo "=== CHAIN COMPLETE $(date '+%F %T %Z') ===" | tee -a "$CHAIN_LOG"
