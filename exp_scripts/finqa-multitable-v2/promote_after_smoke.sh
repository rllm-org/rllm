#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <smoke-pid-file> <smoke-logpath-file> <formal-launcher> <promotion-log>" >&2
  exit 2
fi

smoke_pid_file=$1
smoke_logpath_file=$2
formal_launcher=$3
promotion_log=$4
root_dir=$(cd "$(dirname "$0")" && pwd)

smoke_pid=$(<"$smoke_pid_file")
smoke_log_rel=$(<"$smoke_logpath_file")
if [[ "$smoke_log_rel" = /* ]]; then
  smoke_log=$smoke_log_rel
else
  smoke_log="$root_dir/$smoke_log_rel"
fi

echo "[$(date --iso-8601=seconds)] waiting for smoke pid=$smoke_pid log=$smoke_log" >>"$promotion_log"
while kill -0 "$smoke_pid" 2>/dev/null; do
  sleep 30
done

# Match runtime failures, not harmless configuration keys such as nccl_timeout.
fatal_pattern='CUDA out of memory|OutOfMemoryError|Watchdog caught collective operation timeout|NCCL error|ncclSystemError|RayTaskError|Error executing job|SIGABRT|ValueError: To serve'
if grep -Eiq "$fatal_pattern" "$smoke_log"; then
  echo "[$(date --iso-8601=seconds)] smoke failed: fatal signature found; formal launch blocked" >>"$promotion_log"
  exit 1
fi

if ! grep -Eq 'step:2 -' "$smoke_log"; then
  echo "[$(date --iso-8601=seconds)] smoke incomplete: optimizer step 2 not found; formal launch blocked" >>"$promotion_log"
  exit 1
fi

# Ray/vLLM teardown is asynchronous. Do not overlap two distributed jobs.
for _ in $(seq 1 120); do
  if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    break
  fi
  sleep 10
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "[$(date --iso-8601=seconds)] smoke passed but GPUs did not release; formal launch blocked" >>"$promotion_log"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] smoke passed; launching $formal_launcher" >>"$promotion_log"
exec bash "$formal_launcher" epoch >>"$promotion_log" 2>&1
