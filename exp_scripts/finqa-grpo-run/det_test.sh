#!/usr/bin/env bash
# Does turning off prefix caching + cudagraphs give a deterministic ruler?
# Two runs of the SAME base model, greedy, same seed, same rows.
set -uo pipefail
RD="$(cd "$(dirname "$0")" && pwd)"
for r in D1 D2; do
    rm -rf "$RD/eval/det_$r"
    VLLM_EXTRA="--no-enable-prefix-caching --enforce-eager" \
    EVAL_TEMP=0 EVAL_TOP_P=1.0 PORT_VAL=8150 PORT_TEST=8151 \
        bash "$RD/eval_parallel.sh" base "det_$r" > "$RD/logs/det_$r.log" 2>&1
    echo "=== det_$r done (rc=$?) ==="
done
echo "DET_TEST_DONE"
