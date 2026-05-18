#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/opt/tiger/modelchef}
RLLM_ROOT="$REPO_ROOT/submodules/rllm"
COOKBOOK_DIR="$RLLM_ROOT/cookbooks/swe"
SCRIPT_DIR="$COOKBOOK_DIR/launchers"
export RLLM_ROOT

cd "$REPO_ROOT"

# shellcheck source=swe_artifact_utils.sh
source "$SCRIPT_DIR/swe_artifact_utils.sh"

restore_megatron_cp2_overlay_for_arnold() {
    local artifacts overlay site_packages
    local candidate

    artifacts="$(artifact_dir)"
    overlay="${RLLM_SWE_MEGATRON_OVERLAY:-}"
    if [ -z "$overlay" ]; then
        for candidate in \
            "$artifacts/megatron_core_0.18.0_829a7b78d.tar.gz" \
            /mnt/hdfs/rllm_swe_artifacts/megatron_core_0.18.0_829a7b78d.tar.gz; do
            if [ -s "$candidate" ]; then
                overlay="$candidate"
                break
            fi
        done
    fi

    site_packages="$VIRTUAL_ENV/lib/python3.12/site-packages"
    if [ -n "$overlay" ] && [ -s "$overlay" ]; then
        echo "Restoring Megatron CP2 overlay from $overlay"
        rm -rf "$site_packages/megatron" "$site_packages"/megatron_core-*.dist-info
        tar -C "$site_packages" -xzf "$overlay"
        find "$site_packages/megatron" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
    fi

    "$VIRTUAL_ENV/bin/python" - <<'PY'
from pathlib import Path
import importlib.metadata as md
import os
import sys

required = os.environ.get("RLLM_SWE_REQUIRE_MEGATRON_CP2", "").lower() in {"1", "true", "yes"}
try:
    version = md.version("megatron-core")
except md.PackageNotFoundError:
    version = ""

site = Path(os.environ["VIRTUAL_ENV"]) / "lib/python3.12/site-packages"
tf_config = site / "megatron/core/transformer/transformer_config.py"
text = tf_config.read_text(errors="ignore") if tf_config.exists() else ""
old_assert = "Gated delta net does not support context parallel" in text
has_provider_attr = "overlap_dispatch_backward_with_experts_wgrad" in text
ok = version.startswith("0.18.0") and not old_assert and has_provider_attr
print(
    "megatron_cp2_check "
    f"version={version or '<missing>'} "
    f"old_assert={old_assert} "
    f"has_provider_attr={has_provider_attr} "
    f"ok={ok}",
    flush=True,
)
if required and not ok:
    sys.exit("Megatron CP2 overlay is required but the active package is not CP2-capable")
PY
}

check_rllm_model_gateway_for_arnold() {
    "$VIRTUAL_ENV/bin/python" - <<'PY'
import rllm_model_gateway

print(f"rllm_model_gateway_ok path={rllm_model_gateway.__file__}", flush=True)
PY
}

install_qwen35_mtp_export_hotfix() {
    "$VIRTUAL_ENV/bin/python" - <<'PY'
import os
from pathlib import Path

site = Path(os.environ["VIRTUAL_ENV"]) / "lib/python3.12/site-packages"
path = site / "megatron/bridge/models/conversion/auto_bridge.py"
marker = "rLLM_QWEN35_MTP_EXPORT_HOTFIX"
text = path.read_text()
if marker in text:
    stale = "generator = _rllm_append_original_mtp_tensors(generator, self.hf_pretrained, cpu=cpu)"
    fixed = "generator = _rllm_append_original_mtp_tensors(generator, self.hf_pretrained, cpu=locals().get('cpu', True))"
    if stale in text:
        path.write_text(text.replace(stale, fixed))
        print(f"qwen35_mtp_export_hotfix=updated path={path}", flush=True)
        raise SystemExit(0)
    print("qwen35_mtp_export_hotfix=already_installed", flush=True)
    raise SystemExit(0)

helper = f'''
# {marker}
def _rllm_append_original_mtp_tensors(generator, hf_pretrained, cpu=True):
    yielded = set()
    for name, tensor in generator:
        yielded.add(name)
        yield name, tensor

    try:
        state = hf_pretrained.state
        source = getattr(state, "source", None)
        if source is not None:
            keys = source.get_all_keys()
        else:
            keys = state._get_all_keys()
    except Exception as exc:
        print(f"qwen35_mtp_export_hotfix unable_to_list_mtp_keys: {{exc}}", flush=True)
        return

    mtp_keys = [key for key in keys if key.startswith("mtp.") and key not in yielded]
    if not mtp_keys:
        return

    print(f"qwen35_mtp_export_hotfix appending {{len(mtp_keys)}} original MTP tensors", flush=True)
    for key in mtp_keys:
        try:
            tensor = state[key]
        except Exception as exc:
            print(f"qwen35_mtp_export_hotfix failed_to_load {{key}}: {{exc}}", flush=True)
            continue
        if cpu and hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        yield key, tensor
'''

class_marker = "\n\nclass AutoBridge("
if class_marker not in text:
    raise SystemExit(f"Could not find AutoBridge class marker in {path}")
text = text.replace(class_marker, "\n\n" + helper + class_marker, 1)

needle = "        model_instance = self._get_model_instance(model)\n        quant_tensors = None\n"
replacement = (
    "        generator = _rllm_append_original_mtp_tensors(generator, self.hf_pretrained, cpu=locals().get('cpu', True))\n"
    "        model_instance = self._get_model_instance(model)\n"
    "        quant_tensors = None\n"
)
if needle not in text:
    raise SystemExit(f"Could not find save_hf_weights insertion point in {path}")
path.write_text(text.replace(needle, replacement, 1))
print(f"qwen35_mtp_export_hotfix=installed path={path}", flush=True)
PY
}

install_megatron_checkpoint_json_hotfix() {
    "$VIRTUAL_ENV/bin/python" - <<'PY'
import os
from pathlib import Path

site = Path(os.environ["VIRTUAL_ENV"]) / "lib/python3.12/site-packages"
path = site / "verl/utils/checkpoint/megatron_checkpoint_manager.py"
marker = "rLLM_MEGATRON_CHECKPOINT_JSON_HOTFIX"
text = path.read_text()
if marker in text:
    print("megatron_checkpoint_json_hotfix=already_installed", flush=True)
    raise SystemExit(0)

helper = f'''
# {marker}
def _rllm_json_default(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (torch.dtype, AttnBackend)):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return {{k: v for k, v in vars(obj).items() if not k.startswith("_")}}
    return str(obj)
'''

needle = "\nclass MegatronCheckpointManager(BaseCheckpointManager):\n"
if needle not in text:
    raise SystemExit(f"Could not find checkpoint manager insertion point in {path}")
text = text.replace(needle, "\n" + helper + needle, 1)

old = "                    json.dump(transformer_config_dict, f, indent=2)"
new = "                    json.dump(transformer_config_dict, f, indent=2, default=_rllm_json_default)"
if old not in text:
    raise SystemExit(f"Could not find transformer config json.dump in {path}")
path.write_text(text.replace(old, new, 1))
print(f"megatron_checkpoint_json_hotfix=installed path={path}", flush=True)
PY
}

install_bypass_debug_metrics_hotfix() {
    "$VIRTUAL_ENV/bin/python" - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["RLLM_ROOT"]) / "rllm/experimental/verl/verl_backend.py"
text = path.read_text()
if "old_log_probs_metrics" in text:
    print("bypass_debug_metrics_hotfix=already_installed", flush=True)
    raise SystemExit(0)

old = '''        if bypass_mode:
            assert "rollout_log_probs" in batch.batch, "bypass_mode requires rollout_log_probs in batch"
            with simple_timer("old_log_probs", timing_dict):
                batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        else:
'''
new = '''        if bypass_mode:
            assert "rollout_log_probs" in batch.batch, "bypass_mode requires rollout_log_probs in batch"
            with simple_timer("old_log_probs", timing_dict):
                batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

            # Keep bypass semantics for training, but still run the actor
            # forward pass once so debug metrics match the non-bypass path.
            with simple_timer("old_log_probs_metrics", timing_dict):
                tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
                log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
                entropy = no_padding_2_padding(tu.get(output, "entropy"), batch_td)

                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy"] = entropy_agg.detach().item()

                debug_tensors = {
                    "old_log_probs": log_probs.float(),
                    "rollout_log_probs": batch.batch["rollout_log_probs"],
                    "responses": batch.batch["responses"],
                }
                if "response_mask" in batch.batch:
                    debug_tensors["response_mask"] = batch.batch["response_mask"]
                elif "attention_mask" in batch.batch:
                    debug_tensors["attention_mask"] = batch.batch["attention_mask"]
                metrics.update(calculate_debug_metrics_compat(DataProto.from_dict(tensors=debug_tensors)))
        else:
'''
if old not in text:
    raise SystemExit(f"Could not find bypass metrics insertion point in {path}")
path.write_text(text.replace(old, new, 1))
print(f"bypass_debug_metrics_hotfix=installed path={path}", flush=True)
PY
}

ensure_aiosqlite_for_gateway() {
    local artifacts wheel_dir

    if "$VIRTUAL_ENV/bin/python" - <<'PY'
import aiosqlite

print(f"aiosqlite_ok path={aiosqlite.__file__}", flush=True)
PY
    then
        return 0
    fi

    artifacts="$(artifact_dir)"
    wheel_dir="$artifacts/wheels"
    if compgen -G "$wheel_dir/aiosqlite-*.whl" >/dev/null; then
        "$VIRTUAL_ENV/bin/python" -m pip install --no-index --find-links "$wheel_dir" 'aiosqlite==0.22.1'
    else
        "$VIRTUAL_ENV/bin/python" -m pip install --no-cache-dir -i "${RLLM_SWE_PIP_INDEX_URL:-https://bytedpypi.byted.org/simple}" 'aiosqlite==0.22.1'
    fi

    "$VIRTUAL_ENV/bin/python" - <<'PY'
import aiosqlite

print(f"aiosqlite_ok path={aiosqlite.__file__}", flush=True)
PY
}

echo "============================================================"
echo "rLLM SWE Qwen3.5-9B Megatron Arnold entrypoint"
echo "host=$(hostname) role=${ARNOLD_ROLE:-unknown} task=${ARNOLD_ID:-unknown}"
echo "artifact_dir=$(artifact_dir)"
echo "mode=${RLLM_SWE_MODE:-smoke}"
echo "============================================================"

if [ "${RLLM_SWE_MODE:-smoke}" = "cluster_only" ]; then
    echo "Cluster-only mode: Arnold is holding the Ray cluster for an external SWE driver."
    echo "External driver should connect through the Ray Client port exposed by the current head pod."
    while true; do
        sleep 3600
    done
fi

restore_swe_artifacts
restore_megatron_cp2_overlay_for_arnold
install_qwen35_mtp_export_hotfix
install_megatron_checkpoint_json_hotfix
install_bypass_debug_metrics_hotfix
if [ "${RLLM_SWE_LINK_LOCAL_MODEL_PATH:-1}" = "1" ] \
    && [ ! -d /mnt/hdfs/model_path ] \
    && [ -d /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots ]; then
    MODEL_SNAPSHOT="$(find -L /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots -mindepth 1 -maxdepth 1 -type d -print -quit)"
    if [ -n "$MODEL_SNAPSHOT" ] && [ -d "$MODEL_SNAPSHOT" ]; then
        sudo mkdir -p /mnt/hdfs 2>/dev/null || mkdir -p /mnt/hdfs
        sudo ln -sfn "$MODEL_SNAPSHOT" /mnt/hdfs/model_path 2>/dev/null || ln -sfn "$MODEL_SNAPSHOT" /mnt/hdfs/model_path
    fi
fi
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$COOKBOOK_DIR:$RLLM_ROOT:$RLLM_ROOT/rllm-model-gateway/src:${PYTHONPATH:-}"
export MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/model_path}
export RLLM_SWE_OUTPUT_DIR=${RLLM_SWE_OUTPUT_DIR:-/tmp/rllm_swe_outputs}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$RLLM_SWE_OUTPUT_DIR/checkpoints}
export TRAJ_DIR=${TRAJ_DIR:-$RLLM_SWE_OUTPUT_DIR/trajectories/\${trainer.experiment_name}}
export RLLM_RUN_TASK_RUNNER_LOCAL=1
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export LOGGER=${LOGGER:-"[console,wandb]"}
export SWE_REX_MODAL_APP_LOOKUP_TIMEOUT_S="${SWE_REX_MODAL_APP_LOOKUP_TIMEOUT_S:-60}"
export SWE_REX_MODAL_SANDBOX_CREATE_TIMEOUT_S="${SWE_REX_MODAL_SANDBOX_CREATE_TIMEOUT_S:-240}"
export SWE_REX_MODAL_TUNNELS_TIMEOUT_S="${SWE_REX_MODAL_TUNNELS_TIMEOUT_S:-60}"

ensure_aiosqlite_for_gateway
check_rllm_model_gateway_for_arnold

if [ "${RLLM_SWE_MODE:-smoke}" = "external_ray_worker" ]; then
    if [ -z "${EXTERNAL_RAY_HEAD_IP:-}" ]; then
        echo "Missing EXTERNAL_RAY_HEAD_IP for external_ray_worker mode." >&2
        exit 2
    fi
    nvidia-smi
    "$VIRTUAL_ENV/bin/ray" stop --force || true
    sleep 2
    "$VIRTUAL_ENV/bin/ray" start \
        --address="[${EXTERNAL_RAY_HEAD_IP}]:${EXTERNAL_RAY_HEAD_PORT:-6379}" \
        --num-cpus="${RAY_WORKER_CPUS:-240}" \
        --num-gpus="${RAY_WORKER_GPUS:-8}" \
        --disable-usage-stats
    "$VIRTUAL_ENV/bin/ray" status || true
    echo "EXTERNAL_RAY_WORKER_READY host=$(hostname) role=${ARNOLD_ROLE:-unknown} head=${EXTERNAL_RAY_HEAD_IP}"
    while true; do
        "$VIRTUAL_ENV/bin/ray" status || true
        sleep 300
    done
fi

if [ -z "${MODAL_TOKEN_ID:-}" ] || [ -z "${MODAL_TOKEN_SECRET:-}" ]; then
    echo "Missing MODAL_TOKEN_ID or MODAL_TOKEN_SECRET." >&2
    exit 2
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ "${RLLM_SWE_REQUIRE_WANDB:-1}" = "1" ]; then
    echo "Missing WANDB_API_KEY." >&2
    exit 2
fi

echo "Static network checks"
hostname || true
ip -br addr || true
ip route || true
ip -6 route || true
cat /etc/resolv.conf || true
getent ahosts api.modal.com | head || true
curl -sS \
    --connect-timeout "${RLLM_SWE_MODAL_CURL_CONNECT_TIMEOUT_S:-10}" \
    --max-time "${RLLM_SWE_MODAL_CURL_MAX_TIME_S:-30}" \
    -o /dev/null \
    -w 'remote=%{remote_ip} connect=%{time_connect} total=%{time_total} code=%{http_code}\n' \
    https://api.modal.com || true

cd "$COOKBOOK_DIR"
PROBE_OUT="/tmp/modal_probe_${ARNOLD_TRIAL_ID:-unknown}_${ARNOLD_ID:-0}.jsonl"
SWE_REX_REMOTE_RETRIES=0 python -m swe.scripts.modal_swerex_reliability_test \
    --total "${RLLM_SWE_MODAL_PROBE_TOTAL:-12}" \
    --concurrency "${RLLM_SWE_MODAL_PROBE_CONCURRENCY:-6}" \
    --mode light \
    --out "$PROBE_OUT"

python - "$PROBE_OUT" <<'PY'
import json
import os
import sys

path = sys.argv[1]
expected_total = int(os.environ.get("RLLM_SWE_MODAL_PROBE_TOTAL", "12"))
summary = None
with open(path, encoding="utf-8") as handle:
    for line in handle:
        rec = json.loads(line)
        if rec.get("event") == "summary":
            summary = rec
if summary is None:
    raise SystemExit("BAD_CPU_DRIVER no summary")
p95 = summary.get("create_env_s", {}).get("p95")
p50 = summary.get("create_env_s", {}).get("p50")
ok = (
    summary.get("ok") == summary.get("total") == expected_total
    and summary.get("failed") == 0
    and summary.get("transient_failed") == 0
    and p95 is not None
    and p95 <= 30
    and (p50 is None or p50 <= 60)
)
print("modal_probe_summary=" + json.dumps(summary, sort_keys=True), flush=True)
if not ok:
    raise SystemExit("BAD_CPU_DRIVER")
PY

TRAINING_SCRIPT="$COOKBOOK_DIR/swe/training_scripts/run_swe_training_9b_megatron.sh"
if [ "${RLLM_SWE_MODE:-smoke}" = "smoke" ]; then
    smoke_extra_args=()
    if [ -n "${RLLM_SWE_SMOKE_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        smoke_extra_args=( ${RLLM_SWE_SMOKE_EXTRA_ARGS} )
    fi
    export LOGGER=${LOGGER:-"[console]"}
    export ACTOR_LR_WARMUP_STEPS="${ACTOR_LR_WARMUP_STEPS:-0}"
    export SWE_STEP_LIMIT="${SWE_STEP_LIMIT:-24}"
    export SWE_AGENT_TIMEOUT="${SWE_AGENT_TIMEOUT:-900}"
    export SWE_COMMAND_TIMEOUT="${SWE_COMMAND_TIMEOUT:-120}"
    export SWE_SANDBOX_TIMEOUT="${SWE_SANDBOX_TIMEOUT:-1020}"
    export SWE_STARTUP_JITTER_S="${SWE_STARTUP_JITTER_S:-0}"
    export SWE_VAL_STEP_LIMIT="${SWE_VAL_STEP_LIMIT:-24}"
    export SWE_VAL_AGENT_TIMEOUT="${SWE_VAL_AGENT_TIMEOUT:-900}"
    export SWE_VAL_COMMAND_TIMEOUT="${SWE_VAL_COMMAND_TIMEOUT:-120}"
    export SWE_VAL_SANDBOX_TIMEOUT="${SWE_VAL_SANDBOX_TIMEOUT:-1020}"
    export SWE_VAL_STARTUP_JITTER_S="${SWE_VAL_STARTUP_JITTER_S:-0}"
    exec bash "$TRAINING_SCRIPT" \
        train_max_samples="${RLLM_SWE_SMOKE_TRAIN_SAMPLES:-16}" \
        val_max_samples="${RLLM_SWE_SMOKE_VAL_SAMPLES:-4}" \
        actor_rollout_ref.rollout.n="${RLLM_SWE_SMOKE_ROLLOUT_N:-1}" \
        rllm.workflow.n_parallel_tasks="${RLLM_SWE_SMOKE_PARALLEL_TASKS:-4}" \
        ++trainer.total_training_steps="${RLLM_SWE_SMOKE_STEPS:-1}" \
        trainer.total_epochs=1 \
        trainer.save_freq=1000 \
        trainer.test_freq=1000 \
        "${smoke_extra_args[@]}"
fi

full_extra_args=()
if [ -n "${RLLM_SWE_FULL_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    full_extra_args=( ${RLLM_SWE_FULL_EXTRA_ARGS} )
elif [ -n "${RLLM_SWE_SMOKE_EXTRA_ARGS:-}" ]; then
    # Backward-compatible escape hatch used by the split Arnold CPU driver.
    # shellcheck disable=SC2206
    full_extra_args=( ${RLLM_SWE_SMOKE_EXTRA_ARGS} )
fi

exec bash "$TRAINING_SCRIPT" "${full_extra_args[@]}"
