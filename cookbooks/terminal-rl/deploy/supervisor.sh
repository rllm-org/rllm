#!/bin/bash
# Keeps a SINGLE terminal-RL config training indefinitely (no sweep).
# Use this when you want one long run; use sweep.sh to compare configs.
#
# Handles: process death, disk fill, tunnel expiry, orphaned gateway workers,
# stale listeners on the gateway port.
#
# Config comes from ~/.rllm/terminal-rl-auto.env (mode 600).

set -u

ENV_FILE="${TERMINAL_RL_ENV:-$HOME/.rllm/terminal-rl-auto.env}"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

REPO="${TERMINAL_RL_REPO:-$HOME/rllm}"
CB="$REPO/cookbooks/terminal-rl"
LOGS="${TERMINAL_RL_LOGS:-$HOME/.rllm/terminal-rl-logs}"
EXP="${TERMINAL_RL_EXPERIMENT:-qwen3p5-35b-a3b-tb-v2-debug}"
D="$CB/train_batches/$EXP"
TSTATE="$HOME/.rllm/tunnel.json"
GATEWAY_MODE="${TERMINAL_RL_GATEWAY_MODE:-cloudflared}"
GATEWAY_PORT="${TERMINAL_RL_GATEWAY_PORT:-9200}"

LORA_RANK="${TERMINAL_RL_LORA_RANK:-32}"
LR="${TERMINAL_RL_LR:-8e-5}"
EXTRA_ARGS="${TERMINAL_RL_EXTRA_ARGS:-}"

mkdir -p "$LOGS"
log() { echo "$(date '+%F %T') $*"; }

[ -n "${FIREWORKS_API_KEY:-}" ] || { log "FATAL: FIREWORKS_API_KEY unset (see $ENV_FILE)"; exit 1; }

LOCK="$LOGS/supervisor.pid"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    log "another supervisor is running (pid $(cat "$LOCK")) - exiting"
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

avail_gb() {
    if df -g / >/dev/null 2>&1; then df -g "$REPO" 2>/dev/null | tail -1 | awk '{print $4}'
    else df -BG "$REPO" 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}'; fi
}

last_launch=0
fastfails=0
log "supervisor started (pid $$) rank=$LORA_RANK lr=$LR gateway=$GATEWAY_MODE"

while true; do
    # ---- prune trained-through episode logs (files only; removing empty dirs
    # kills the logger mid-run with FileNotFoundError)
    if [ -d "$D" ]; then
        avail=$(avail_gb); [ -z "$avail" ] && avail=99
        age=20; [ "$avail" -lt 25 ] && age=5; [ "$avail" -lt 12 ] && age=1
        n=$(find "$D" -type f -mmin +$age 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n" -gt 0 ]; then
            find "$D" -type f -mmin +$age -delete 2>/dev/null
            log "pruned $n files (age>${age}m) avail=${avail}G"
        fi
    fi

    # ---- alive? pgrep can fail outright; fall back to ps before relaunching
    if pgrep -f "train_debug.py" >/dev/null 2>&1; then sleep 60; continue; fi
    rc=$?
    if [ "$rc" -gt 1 ] && ps -eo command 2>/dev/null | grep -q "[t]rain_debug.py"; then
        sleep 60; continue
    fi

    now=$(date +%s)
    if [ $((now - last_launch)) -lt 240 ]; then sleep 30; continue; fi
    if [ "$last_launch" -ne 0 ]; then
        ran=$((now - last_launch))
        if [ "$ran" -lt 600 ]; then fastfails=$((fastfails + 1)); else fastfails=0; fi
        log "training down after ${ran}s (consecutive fast failures: $fastfails)"
        if [ "$fastfails" -ge 5 ]; then
            log "5 fast failures in a row - backing off 15m"
            sleep 900; fastfails=0; continue
        fi
    fi

    # ---- gateway URL
    if [ "$GATEWAY_MODE" = "public" ]; then
        URL="${TERMINAL_RL_PUBLIC_URL:-}"
        [ -n "$URL" ] || { log "FATAL: TERMINAL_RL_PUBLIC_URL unset"; sleep 300; continue; }
    else
        URL=$(python3 -c "import json;print(json.load(open('$TSTATE'))['url'])" 2>/dev/null)
        TPID=$(python3 -c "import json;print(json.load(open('$TSTATE'))['pid'])" 2>/dev/null)
        ok=0
        if [ -n "${TPID:-}" ] && kill -0 "$TPID" 2>/dev/null && [ -n "${URL:-}" ] \
           && [ -n "$(dig +short "${URL#https://}" 2>/dev/null)" ]; then ok=1; fi
        if [ "$ok" -ne 1 ]; then
            log "tunnel dead - recreating"
            rllm tunnel down >/dev/null 2>&1
            rllm tunnel up --port "$GATEWAY_PORT" >/dev/null 2>&1
            sleep 8
            URL=$(python3 -c "import json;print(json.load(open('$TSTATE'))['url'])" 2>/dev/null)
            if [ -z "${URL:-}" ] || [ -z "$(dig +short "${URL#https://}" 2>/dev/null)" ]; then
                log "tunnel unavailable - retry in 3m"; sleep 180; continue
            fi
            log "new tunnel: $URL"
        fi
    fi

    # ---- clear orphaned gateway workers; they outlive their parent and would
    # make the next run's episodes come back with no traces (silently filtered)
    if pgrep -f "rllm_model_gateway" >/dev/null 2>&1; then
        n=$(pgrep -f "rllm_model_gateway" | wc -l | tr -d ' ')
        log "killing $n orphaned gateway worker(s)"
        pkill -f "rllm_model_gateway" 2>/dev/null; sleep 5
        pkill -9 -f "rllm_model_gateway" 2>/dev/null
    fi
    for p in $(lsof -nP -iTCP:"$GATEWAY_PORT" -sTCP:LISTEN -t 2>/dev/null); do
        log "killing stale listener on $GATEWAY_PORT (pid $p)"; kill "$p" 2>/dev/null
    done
    sleep 3

    RUNLOG="$LOGS/run_$(date +%Y%m%d_%H%M%S).log"
    cd "$CB" || { log "cannot cd $CB"; sleep 60; continue; }
    # shellcheck disable=SC2086
    nohup bash train_fireworks_debug.sh \
        rllm.gateway.tunnel="$URL" \
        rllm.gateway.port="$GATEWAY_PORT" \
        model.lora_rank="$LORA_RANK" \
        training.learning_rate="$LR" \
        rllm.trainer.save_freq=10 \
        $EXTRA_ARGS > "$RUNLOG" 2>&1 &
    last_launch=$(date +%s)
    log "launched training pid=$! log=$RUNLOG url=$URL"
    ln -sfn "$RUNLOG" "$LOGS/current_run.log"
    ls -t "$LOGS"/run_*.log 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null
    sleep 60
done
