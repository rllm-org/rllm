#!/bin/bash
# Sequential hyperparameter sweep for terminal-RL, portable across macOS and
# Linux. Runs ONE training job at a time.
#
# Each config trains to TARGET_STEPS optimizer steps. A crash mid-config RESUMES
# from the last DCP checkpoint (save_freq=10) rather than restarting from base
# weights, and progress is measured as resumed-step + steps-since, so the number
# reflects consecutive training rather than a sum of independent attempts. Each
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
# Cartesian product RANKS x LRS, run sequentially (one trainer job at a time).
# NOTE: lora_alpha is a fixed 32 unless overridden, so changing rank also changes
# the scaling alpha/r (1.0 at r32, 0.25 at r128). To vary capacity while holding
# scaling constant, set +model.lora_alpha=2*rank per config instead.
RANKS="${TERMINAL_RL_RANKS:-${TERMINAL_RL_LORA_RANK:-128}}"
TARGET_STEPS="${TERMINAL_RL_TARGET_STEPS:-150}"
LRS="${TERMINAL_RL_LRS:-5e-5 1.5e-4}"
EXTRA_ARGS="${TERMINAL_RL_EXTRA_ARGS:-}"   # e.g. "+model.lora_alpha=256"
# Rollout concurrency. The script ships 192, which saturates a 2-replica
# rollout deployment: observed 2815 permanent "429 Too Many Requests" sampling
# failures across 2418 episodes -- more than one per episode -- dragging reward
# well below the ~0.45-0.50 baseline. Those are inference-side 429s from
# api.fireworks.ai, not tunnel throttling. Lower further if 429s persist.
N_PARALLEL="${TERMINAL_RL_N_PARALLEL:-96}"

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
    local rank="$1" lr="$2" name alpha
    name="$(slug "$EXP_BASE")-r${rank}-lr$(slug "$lr")"
    alpha=$(printf '%s' "$EXTRA_ARGS" | grep -oE 'lora_alpha=[0-9]+' | cut -d= -f2)
    [ -n "$alpha" ] && name="${name}-a${alpha}"
    [ -n "$HOST_TAG" ] && name="${name}-$(slug "$HOST_TAG")"
    printf '%s' "$name"
}

cfg_logs()   { printf '%s' "$LOGS/sweep_r$1_lr$2_"; }
base_file()  { printf '%s' "$LOGS/.base_r$1_lr$2"; }

# Highest DCP checkpoint a log reached; snapshots are named "step-<n>".
last_ckpt()  { grep -oE "DCP checkpoint saved: step-[0-9]+" "$1" 2>/dev/null \
                 | grep -oE "[0-9]+$" | sort -n | tail -1; }

# Progress = where the resumed-from checkpoint left off, plus steps taken since.
# Do NOT sum optimizer steps across attempts: without resume each attempt starts
# from base weights, so a config that died at 100 and reran 50 would report 150
# while the final model had only ever seen 50 consecutive steps.
steps_for() {
    local base newest
    base=$(cat "$(base_file "$1" "$2")" 2>/dev/null); base=${base:-0}
    newest=$(ls -t "$(cfg_logs "$1" "$2")"*.log 2>/dev/null | head -1)
    [ -n "$newest" ] || { echo "$base"; return; }
    echo $(( base + $(grep -c "time/optim_step" "$newest" 2>/dev/null) ))
}
eps_for()    { cat "$(cfg_logs "$1" "$2")"*.log 2>/dev/null | grep -c "Rewards:"; }
ones_for()   { cat "$(cfg_logs "$1" "$2")"*.log 2>/dev/null | grep -o "mini-swe-agent: 1.0" | wc -l | tr -d ' '; }

# Resume the next attempt from the last checkpoint of the previous one.
# Checkpoints survive job deletion (archived jobs retain them), so this works
# even after the sweep cleans up the trainer job.
resume_args() {
    local rank="$1" lr="$2" prev ckpt job
    prev=$(ls -t "$(cfg_logs "$rank" "$lr")"*.log 2>/dev/null | head -1)
    [ -n "$prev" ] || return 0
    ckpt=$(last_ckpt "$prev")
    [ -n "$ckpt" ] || return 0
    job=$(grep -oE "training-api-service-[0-9a-f]+" "$prev" 2>/dev/null | head -1)
    [ -n "$job" ] || return 0
    printf 'training.resume_from_dcp_checkpoint=step-%s training.resume_from_fireworks_job_id=%s' "$ckpt" "$job"
}

log "sweep starting: ranks=[$RANKS] lrs=[$LRS] target=$TARGET_STEPS gateway=$GATEWAY_MODE extra='$EXTRA_ARGS'"

for LORA_RANK in $RANKS; do
for LR in $LRS; do
    cfg_start=$(date +%s)
    attempts=0
    jobs=""
    note=ok

    while :; do
        steps=$(steps_for "$LORA_RANK" "$LR")
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
        RESUME="$(resume_args "$LORA_RANK" "$LR")"
        if [ -n "$RESUME" ]; then
            NEW_BASE=$(printf '%s' "$RESUME" | grep -oE 'dcp_checkpoint=step-[0-9]+' | grep -oE '[0-9]+$')
            echo "${NEW_BASE:-0}" > "$(base_file "$LORA_RANK" "$LR")"
            log "r$LORA_RANK lr=$LR resuming from step-$NEW_BASE"
        else
            echo 0 > "$(base_file "$LORA_RANK" "$LR")"
            [ "$attempts" -gt 1 ] && log "r$LORA_RANK lr=$LR no checkpoint to resume - restarting from base weights"
        fi
        RUNLOG="$LOGS/sweep_r${LORA_RANK}_lr${LR}_$(date +%Y%m%d_%H%M%S).log"
        EXP_NAME="$(experiment_name "$LORA_RANK" "$LR")"
        cd "$CB" || { log "cannot cd $CB"; exit 1; }
        # shellcheck disable=SC2086
        nohup bash train_fireworks_debug.sh \
            rllm.gateway.tunnel="$URL" \
            rllm.gateway.port="$GATEWAY_PORT" \
            model.lora_rank="$LORA_RANK" \
            training.learning_rate="$LR" \
            rllm.workflow.n_parallel_tasks="$N_PARALLEL" \
            $RESUME \
        rllm.trainer.save_freq=10 \
            rllm.trainer.experiment_name="$EXP_NAME" \
            rllm.episode_logging.episode_log_dir="train_batches/$EXP_NAME" \
            rllm.episode_logging.backend_batch_log_dir="train_batches/$EXP_NAME" \
            $EXTRA_ARGS > "$RUNLOG" 2>&1 &
        log "r$LORA_RANK lr=$LR attempt $attempts/$MAX_ATTEMPTS exp=$EXP_NAME (steps so far: $steps/$TARGET_STEPS) log=$RUNLOG"
        ln -sfn "$RUNLOG" "$LOGS/current_run.log"

        launched=$(date +%s)
        placed=no
        while :; do
            sleep 60
            prune_logs

            if [ "$placed" = no ] && grep -q "healthz=OK" "$RUNLOG" 2>/dev/null; then
                placed=yes
                log "r$LORA_RANK lr=$LR placed after $(( ($(date +%s) - launched) / 60 ))m"
            fi
            if [ "$placed" = no ] && [ $(( ($(date +%s) - launched) / 60 )) -ge "$PLACE_LIMIT_MIN" ]; then
                log "r$LORA_RANK lr=$LR never placed in ${PLACE_LIMIT_MIN}m - abandoning attempt"
                break
            fi
            [ "$(steps_for "$LORA_RANK" "$LR")" -ge "$TARGET_STEPS" ] && { log "r$LORA_RANK lr=$LR reached $TARGET_STEPS steps"; break; }
            training_alive || { log "r$LORA_RANK lr=$LR run exited (steps now $(steps_for "$LORA_RANK" "$LR"))"; break; }
            [ $(( ($(date +%s) - cfg_start) / 3600 )) -ge "$MAX_HOURS" ] && break
        done

        J=$(grep -oE "training-api-service-[0-9a-f]+" "$RUNLOG" 2>/dev/null | head -1)
        [ -n "$J" ] && jobs="$jobs $J"
        stop_training
        [ -n "$J" ] && delete_job "$J"
    done

    S=$(steps_for "$LORA_RANK" "$LR"); E=$(eps_for "$LORA_RANK" "$LR"); O=$(ones_for "$LORA_RANK" "$LR")
    RM=$(python3 -c "print(f'{$O/$E:.3f}' if $E else 'n/a')" 2>/dev/null)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date '+%F %T')" "$LR" "$LORA_RANK" "$S" "$TARGET_STEPS" "$E" "$RM" "$attempts" "${jobs# }" "$note" >> "$RESULTS"
    log "r$LORA_RANK lr=$LR FINISHED: steps=$S/$TARGET_STEPS episodes=$E reward_mean=$RM attempts=$attempts note=$note"
done
done

log "sweep complete -> $RESULTS"
stop_training
