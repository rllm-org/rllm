#!/usr/bin/env bash
# Atomically stage the already-merged PRPO models that live only on .29 disk1t.
set -euo pipefail

HOST=10.225.68.16
SOURCE=/mnt/disk1t/finqa-prpo-run-checkpoints/merged-eval-models
DEST=/home/yanan/.cache/finqa-multitable-v2-merged-cache

# Interleave the two evaluation lanes so neither GPU waits for its next model.
ITEMS=(
    global_step_64 global_step_128 global_step_192 global_step_368
    global_step_304 global_step_496 global_step_432 global_step_256
    global_step_560 global_step_288 global_step_240 global_step_336
    global_step_272 global_step_384 global_step_320 global_step_416
    global_step_352 global_step_448 global_step_400 global_step_480
    global_step_464 global_step_528 global_step_512 global_step_576
    global_step_544 global_step_608 global_step_592
)

ssh "$HOST" "mkdir -p '$DEST'"
for item in "${ITEMS[@]}"; do
    src="$SOURCE/$item"
    dst="$DEST/prpo_$item"
    [[ -f "$src/config.json" ]] || { echo "missing merged source: $src" >&2; exit 1; }
    find "$src" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q . || {
        echo "merged source lacks weights: $src" >&2
        exit 1
    }
    if ssh "$HOST" "test -f '$dst/config.json' && find '$dst' -maxdepth 1 -name '*.safetensors' -print -quit | grep -q ."; then
        echo "[stage] already complete: $item"
        continue
    fi
    tmp="${dst}.incoming"
    ssh "$HOST" "rm -rf '$tmp' && mkdir -p '$tmp'"
    echo "[stage] $item"
    rsync -a --partial "$src/" "$HOST:$tmp/"
    ssh "$HOST" "test -f '$tmp/config.json' && \
        find '$tmp' -maxdepth 1 -name '*.safetensors' -print -quit | grep -q . && \
        mv '$tmp' '$dst'"
done

echo PRPO_EARLY_STAGING_COMPLETE
