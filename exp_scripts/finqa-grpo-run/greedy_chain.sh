#!/usr/bin/env bash
# Wait for greedy_G2, apply the PRE-REGISTERED reproducibility criterion, and
# only then evaluate the checkpoints.
#
# Criterion, fixed before the data was seen (|G1 - G2| in tasks, on BOTH splits):
#   0        -> greedy is deterministic; any delta >= 1 task is signal
#   1..3     -> usable; noise floor ~0.2-0.6 pt, only deltas above ~1 pt count
#   >3       -> greedy does not fix it; stop and report, do not run checkpoints
#
# base is NOT re-run: greedy_G1 and greedy_G2 ARE the base measurement under
# greedy, and having two of them gives base an error bar the checkpoints lack.
set -uo pipefail
RD="$(cd "$(dirname "$0")" && pwd)"
GATE=3

for _ in $(seq 1 90); do
    [ -f "$RD/eval/greedy_G2/val.json" ] && [ -f "$RD/eval/greedy_G2/test.json" ] && break
    sleep 60
done

read -r VERDICT DV DT < <(python3 - <<'PY'
import json, pathlib
RD = pathlib.Path("/home/yanan/agents/rllm/exp_scripts/finqa-grpo-run/eval")
def c(r, s):
    f = RD / r / f"{s}.json"
    return json.load(open(f))["correct"] if f.exists() else None
dv = dt = None
try:
    dv = abs(c("greedy_G1", "val") - c("greedy_G2", "val"))
    dt = abs(c("greedy_G1", "test") - c("greedy_G2", "test"))
    print(("PASS" if max(dv, dt) <= 3 else "FAIL"), dv, dt)
except TypeError:
    print("INCOMPLETE", -1, -1)
PY
)

echo "GREEDY_GATE verdict=$VERDICT delta_val=$DV delta_test=$DT"
if [ "$VERDICT" != "PASS" ]; then
    echo "GREEDY_GATE: not running checkpoint evals ($VERDICT)."
    exit 1
fi

for CK in global_step_31 global_step_62; do
    echo "=== greedy eval: $CK ==="
    rm -rf "$RD/eval/greedy_$CK"
    PORT_VAL=8140 PORT_TEST=8141 \
        bash "$RD/eval_parallel.sh" "$RD/merged/$CK" "greedy_$CK" \
        > "$RD/logs/greedy_$CK.log" 2>&1
    echo "=== done: $CK (rc=$?) ==="
done
echo "GREEDY_CHAIN_DONE"
