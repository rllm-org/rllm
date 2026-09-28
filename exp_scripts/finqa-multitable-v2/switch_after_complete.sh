#!/usr/bin/env bash
# Replace an already-running broad queue only after its current checkpoint has
# completed both splits and written the protocol audit marker.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OLD_PID="${1:?old eval_lane pid}"
CURRENT_COMPLETE="${2:?current COMPLETE marker}"
ARM="${3:?arm}"
GPU="${4:?gpu}"
PORT="${5:?port}"
shift 5
[[ "$#" -gt 0 ]] || { echo "no replacement checkpoints" >&2; exit 2; }

while [[ ! -e "$CURRENT_COMPLETE" ]]; do
    kill -0 "$OLD_PID" 2>/dev/null || {
        echo "old lane $OLD_PID exited before $CURRENT_COMPLETE" >&2
        exit 1
    }
    sleep 2
done

# Resolve the exact descendants before sending signals.  This prevents the old
# lane from beginning another checkpoint from its original broad argv queue.
python - "$OLD_PID" <<'PY'
import os, signal, sys
root = int(sys.argv[1])
parents = {}
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    try:
        parents[int(entry)] = int(open(f"/proc/{entry}/stat").read().split()[3])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
selected = {root}
while True:
    expanded = selected | {pid for pid, ppid in parents.items() if ppid in selected}
    if expanded == selected:
        break
    selected = expanded
for pid in sorted(selected, reverse=True):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
PY

exec bash "$DIR/eval_lane.sh" "$ARM" "$GPU" "$PORT" "$@"
