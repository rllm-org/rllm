#!/bin/bash
# Sequential hyperparameter sweep for terminal-RL, portable across macOS and
# Linux. Runs ONE training job at a time.
#
# Each config accumulates TARGET_STEPS optimizer steps, counted across restarts,
# so a crash mid-config resumes the count instead of starting over. Each
# config's trainer job -- and only its own -- is deleted when the config ends,
# so nothing is left billing on a shared account.
#
# Guards (all cost-motivated, from real failures):
#   * placement:  an attempt that never reports healthz=OK within
#                 PLACE_LIMIT_MIN is abandoned. Trainer jobs have sat in
#                 CREATING for a full hour and produced nothing.
#   * attempts:   at most MAX_ATTEMPTS launches per config.
#   * wall clock: at most MAX_HOURS per config regardless of progress.
#
# Config comes from ~/.rllm/terminal-rl-auto.env (mode 600). See
# terminal-rl.env.example.

set -u

ENV_FILE="${TERMINAL_RL_ENV:-$HOME/.rllm/terminal-rl-auto.env}"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

REPO="${TERMINAL_RL_REPO:-$HOME/rllm}"
CB="$REPO/cookbooks/terminal-rl"
LOGS="${TERMINAL_RL_LOGS:-$HOME/.rllm/terminal-rl-logs}"
EXP_BASE="${TERMINAL_RL_EXPERIMENT:-qwen3p5-35b-a3b-tb-v2-debug}"
# Prune the whole tree: each config now writes to its own per-experiment dir.
D="$CB/train_batches"
# Optional suffix to distinguish coordinators when two hosts share an account.
HOST_TAG="${TERMINAL_RL_HOST_TAG:-}"
TSTATE="$HOME/.rllm/tunnel.json"
RESULTS="$LOGS/sweep_results.tsv"
ACCT="${TERMINAL_RL_ACCOUNT:-research}"
GATEWAY_MODE="${TERMINAL_RL_GATEWAY_MODE:-cloudflared}"
GATEWAY_PORT="${TERMINAL_RL_GATEWAY_PORT:-9200}"

# ---- sweep grid -------------------------------------------------------------
LORA_RANK="${TERMINAL_RL_LORA_RANK:-128}"
TARGET_STEPS="${TERMINAL_RL_TARGET_STEPS:-150}"
LRS="${TERMINAL_RL_LRS:-5e-5 1.5e-4}"
EXTRA_ARGS="${TERMINAL_RL_EXTRA_ARGS:-}"   # e.g. "+model.lora_alpha=256"

PLACE_LIMIT_MIN="${TERMINAL_RL_PLACE_LIMIT_MIN:-75}"
MAX_ATTEMPTS="${TERMINAL_RL_MAX_ATTEMPTS:-4}"
MAX_HOURS="${TERMINAL_RL_MAX_HOURS:-60}"

mkdir -p "$LOGS"
log() { echo "$(date '+%F %T') [sweep] $*"; }

if [ -z "${FIREWORKS_API_KEY:-}" ]; then
    log "FATAL: FIREWORKS_API_KEY unset (see $ENV_FILE)"
    exit 1
fi

# Single-instance guard: a service manager plus a manual start would otherwise
# run two sweeps, two gateways, and two billing trainer jobs.
LOCK="$LOGS/sweep.pid"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    log "another sweep is running (pid $(cat "$LOCK")) - exiting"
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

[ -f "$RESULTS" ] || printf "finished\tlr\trank\tsteps\ttarget\tepisodes\treward_mean\tattempts\tjobs\tnote\n" > "$RESULTS"

# df -g is macOS; -BG is GNU. Report available GiB on whichever is present.
avail_gb() {
    if df -g / >/dev/null 2>&1; then
        df -g "$REPO" 2>/dev/null | tail -1 | awk '{print $4}'
    else
        df -BG "$REPO" 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}'
    fi
}

stop_training() {
    pkill -f "train_debug.py" 2>/dev/null
    pkill -f "train_fireworks_debug.sh" 2>/dev/null
    sleep 8
    pkill -9 -f "train_debug.py" 2>/dev/null
    # Gateway workers are separate processes that OUTLIVE their parent. Left
    # alive they keep the gateway port, and the next run's episodes come back
    # "No traces found" and are silently filtered -- a run once burned 1h50m
    # and 383 episodes this way while looking perfectly healthy.
    pkill -f "rllm_model_gateway" 2>/dev/null
    sleep 4
    pkill -9 -f "rllm_model_gateway" 2>/dev/null
    for p in $(lsof -nP -iTCP:"$GATEWAY_PORT" -sTCP:LISTEN -t 2>/dev/null); do kill "$p" 2>/dev/null; done
    sleep 2
}

# Deletes ONLY ids this sweep created. Never widen this to "all active jobs":
# the account is shared and that would kill a colleague's training run.
delete_job() {
    local jid="$1" code
    case "$jid" in
        training-api-service-*) ;;
        *) log "refusing to delete suspicious id '$jid'"; return 1;;
    esac
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -X DELETE \
        -H "Authorization: Bearer $FIREWORKS_API_KEY" \
        "https://api.fireworks.ai/v1/accounts/$ACCT/rlorTrainerJobs/$jid")
    log "deleted $jid (HTTP $code)"
}

# Sets URL. On a server prefer public mode: quick tunnels expire, get
# rate-limited, and are blocked by some DNS filters.
ensure_gateway_url() {
    if [ "$GATEWAY_MODE" = "public" ]; then
        URL="${TERMINAL_RL_PUBLIC_URL:-}"
        [ -n "$URL" ] || { log "FATAL: TERMINAL_RL_PUBLIC_URL unset in public mode"; return 1; }
        return 0
    fi
    URL=$(python3 -c "import json;print(json.load(open('$TSTATE'))['url'])" 2>/dev/null)
    TPID=$(python3 -c "import json;print(json.load(open('$TSTATE'))['pid'])" 2>/dev/null)
    if [ -n "${TPID:-}" ] && kill -0 "$TPID" 2>/dev/null && [ -n "${URL:-}" ] \
       && [ -n "$(dig +short "${URL#https://}" 2>/dev/null)" ]; then
        return 0
    fi
    log "tunnel dead - recreating"
    rllm tunnel down >/dev/null 2>&1
    rllm tunnel up --port "$GATEWAY_PORT" >/dev/null 2>&1
    sleep 8
    URL=$(python3 -c "import json;print(json.load(open('$TSTATE'))['url'])" 2>/dev/null)
    [ -n "${URL:-}" ] && [ -n "$(dig +short "${URL#https://}" 2>/dev/null)" ]
}

prune_logs() {
    [ -d "$D" ] || return 0
    local avail age
    avail=$(avail_gb); [ -z "$avail" ] && avail=99
    age=20; [ "$avail" -lt 25 ] && age=5; [ "$avail" -lt 12 ] && age=1
    # Files ONLY. Never `-type d -empty -delete`: the logger writes into
    # episodes/ and backend_batches/ and dies with FileNotFoundError if the
    # directory disappears between steps (this killed a run at step 10).
    find "$D" -type f -mmin +$age -delete 2>/dev/null
}

training_alive() {
    if pgrep -f "train_debug.py" >/dev/null 2>&1; then return 0; fi
    # pgrep can fail outright (macOS sysmond entitlement loss). Treat an error
    # as "assume alive" via a ps fallback, so a broken check never causes a
    # duplicate launch.
    [ $? -gt 1 ] && ps -eo command 2>/dev/null | grep -q "[t]rain_debug.py" && return 0
    return 1
}

# Fireworks builds promoted-model ids as "<experiment_name>-step-<n>", so the
# experiment name has to be id-safe: lowercase, hyphens, no dots
# (1.5e-4 -> 1p5e-4). Encoding rank/lr/alpha in the name also keeps two
# coordinators from overwriting each other's promoted checkpoints, and keeps
# wandb runs and episode-log dirs separate per config.
slug() { printf '%s' "$1" | tr 'A-Z' 'a-z' | tr '.' 'p' | tr -cd 'a-z0-9-'; }

experiment_name() {
    local lr="$1" name alpha
    name="$(slug "$EXP_BASE")-r${LORA_RANK}-lr$(slug "$lr")"
    alpha=$(printf '%s' "$EXTRA_ARGS" | grep -oE 'lora_alpha=[0-9]+' | cut -d= -f2)
    [ -n "$alpha" ] && name="${name}-a${alpha}"
    [ -n "$HOST_TAG" ] && name="${name}-$(slug "$HOST_TAG")"
    printf '%s' "$name"
}

steps_for_lr() { cat "$LOGS"/sweep_lr"$1"_*.log 2>/dev/null | grep -c "time/optim_step"; }
eps_for_lr()   { cat "$LOGS"/sweep_lr"$1"_*.log 2>/dev/null | grep -c "Rewards:"; }
ones_for_lr()  { cat "$LOGS"/sweep_lr"$1"_*.log 2>/dev/null | grep -o "mini-swe-agent: 1.0" | wc -l | tr -d ' '; }

log "sweep starting: rank=$LORA_RANK target=$TARGET_STEPS lrs=[$LRS] gateway=$GATEWAY_MODE extra='$EXTRA_ARGS'"

for LR in $LRS; do
    cfg_start=$(date +%s)
    attempts=0
    jobs=""
    note=ok

    while :; do
        steps=$(steps_for_lr "$LR")
        [ "$steps" -ge "$TARGET_STEPS" ] && { note=ok; break; }
        [ "$attempts" -ge "$MAX_ATTEMPTS" ] && { note="gave-up-after-${attempts}-attempts"; break; }
        [ $(( ($(date +%s) - cfg_start) / 3600 )) -ge "$MAX_HOURS" ] && { note="wall-clock-${MAX_HOURS}h"; break; }

        stop_training
        if ! ensure_gateway_url; then
            log "no gateway URL - waiting 5m"
            sleep 300
            continue
        fi

        attempts=$((attempts + 1))
        RUNLOG="$LOGS/sweep_lr${LR}_$(date +%Y%m%d_%H%M%S).log"
        EXP_NAME="$(experiment_name "$LR")"
        cd "$CB" || { log "cannot cd $CB"; exit 1; }
        # shellcheck disable=SC2086
        nohup bash train_fireworks_debug.sh \
            rllm.gateway.tunnel="$URL" \
            rllm.gateway.port="$GATEWAY_PORT" \
            model.lora_rank="$LORA_RANK" \
            training.learning_rate="$LR" \
            rllm.trainer.save_freq=10 \
            rllm.trainer.experiment_name="$EXP_NAME" \
            rllm.episode_logging.episode_log_dir="train_batches/$EXP_NAME" \
            rllm.episode_logging.backend_batch_log_dir="train_batches/$EXP_NAME" \
            $EXTRA_ARGS > "$RUNLOG" 2>&1 &
        log "lr=$LR attempt $attempts/$MAX_ATTEMPTS exp=$EXP_NAME (steps so far: $steps/$TARGET_STEPS) log=$RUNLOG"
        ln -sfn "$RUNLOG" "$LOGS/current_run.log"

        launched=$(date +%s)
        placed=no
        while :; do
            sleep 60
            prune_logs

            if [ "$placed" = no ] && grep -q "healthz=OK" "$RUNLOG" 2>/dev/null; then
                placed=yes
                log "lr=$LR placed after $(( ($(date +%s) - launched) / 60 ))m"
            fi
            if [ "$placed" = no ] && [ $(( ($(date +%s) - launched) / 60 )) -ge "$PLACE_LIMIT_MIN" ]; then
                log "lr=$LR never placed in ${PLACE_LIMIT_MIN}m - abandoning attempt"
                break
            fi
            [ "$(steps_for_lr "$LR")" -ge "$TARGET_STEPS" ] && { log "lr=$LR reached $TARGET_STEPS steps"; break; }
            training_alive || { log "lr=$LR run exited (steps now $(steps_for_lr "$LR"))"; break; }
            [ $(( ($(date +%s) - cfg_start) / 3600 )) -ge "$MAX_HOURS" ] && break
        done

        J=$(grep -oE "training-api-service-[0-9a-f]+" "$RUNLOG" 2>/dev/null | head -1)
        [ -n "$J" ] && jobs="$jobs $J"
        stop_training
        [ -n "$J" ] && delete_job "$J"
    done

    S=$(steps_for_lr "$LR"); E=$(eps_for_lr "$LR"); O=$(ones_for_lr "$LR")
    RM=$(python3 -c "print(f'{$O/$E:.3f}' if $E else 'n/a')" 2>/dev/null)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date '+%F %T')" "$LR" "$LORA_RANK" "$S" "$TARGET_STEPS" "$E" "$RM" "$attempts" "${jobs# }" "$note" >> "$RESULTS"
    log "lr=$LR FINISHED: steps=$S/$TARGET_STEPS episodes=$E reward_mean=$RM attempts=$attempts note=$note"
done

log "sweep complete -> $RESULTS"
stop_training
