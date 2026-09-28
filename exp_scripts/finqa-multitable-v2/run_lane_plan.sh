#!/usr/bin/env bash
# Fixed, auditable full re-evaluation plan. Integer/near-integer epochs run first.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:?usage: $0 0|1 PORT}"
PORT="${2:?missing port}"

if [[ "$GPU" == 0 ]]; then
    GRPO=(
        base global_step_124 global_step_372 global_step_620
        global_step_31 global_step_93 global_step_155 global_step_217
        global_step_279 global_step_341 global_step_403 global_step_465
    )
    PRPO=(
        global_step_64 global_step_192 global_step_304 global_step_432
        global_step_560 global_step_682 global_step_806 global_step_930
        global_step_1054 global_step_1178 global_step_1302
        global_step_240 global_step_272 global_step_320 global_step_352
        global_step_400 global_step_464 global_step_512 global_step_544
        global_step_592
    )
elif [[ "$GPU" == 1 ]]; then
    GRPO=(
        global_step_248 global_step_496
        global_step_62 global_step_186 global_step_310 global_step_434
        global_step_527 global_step_558 global_step_589 global_step_651
        global_step_682 global_step_713
    )
    PRPO=(
        global_step_128 global_step_368 global_step_496 global_step_620
        global_step_744 global_step_868 global_step_992 global_step_1116
        global_step_1240
        global_step_256 global_step_288 global_step_336 global_step_384
        global_step_416 global_step_448 global_step_480 global_step_528
        global_step_576 global_step_608
    )
else
    echo "GPU must be 0 or 1" >&2
    exit 2
fi

bash "$DIR/eval_lane.sh" grpo "$GPU" "$PORT" "${GRPO[@]}"
bash "$DIR/eval_lane.sh" prpo "$GPU" "$PORT" "${PRPO[@]}"
echo "V2_PLAN_COMPLETE gpu=$GPU"
