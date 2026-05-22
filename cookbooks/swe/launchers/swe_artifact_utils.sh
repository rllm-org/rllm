#!/usr/bin/env bash
set -euo pipefail

artifact_dir() {
    local candidate
    if [ -n "${RLLM_SWE_ARTIFACT_DIR:-}" ]; then
        echo "$RLLM_SWE_ARTIFACT_DIR"
        return
    fi
    for candidate in \
        /opt/tiger/swe_runtime_image/artifacts \
        /mnt/hdfs/rllm_swe_artifacts; do
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
    echo /mnt/hdfs/rllm_swe_artifacts
}

setup_b200_driver_libs() {
    local real_libcuda=""
    local real_nvml_lib=""
    local driver_lib_dir=/tmp/rllm_swe_driver_libs
    local f
    for f in /usr/lib/x86_64-linux-gnu/libcuda.so.*; do
        if [ -f "$f" ] && [ -s "$f" ] && [ ! -L "$f" ]; then
            real_libcuda="$f"
        fi
    done
    for f in /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*; do
        if [ -f "$f" ] && [ -s "$f" ] && [ ! -L "$f" ]; then
            real_nvml_lib="$f"
        fi
    done
    mkdir -p "$driver_lib_dir"
    if [ -n "$real_libcuda" ]; then
        ln -sfn "$real_libcuda" "$driver_lib_dir/libcuda.so.1"
        ln -sfn "$real_libcuda" "$driver_lib_dir/libcuda.so"
    fi
    if [ -n "$real_nvml_lib" ]; then
        ln -sfn "$real_nvml_lib" "$driver_lib_dir/libnvidia-ml.so.1"
        ln -sfn "$real_nvml_lib" "$driver_lib_dir/libnvidia-ml.so"
    fi

    export LD_LIBRARY_PATH="$driver_lib_dir:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    if [ -f "$driver_lib_dir/libcuda.so.1" ]; then
        if [ -n "$real_nvml_lib" ]; then
            export LD_PRELOAD="$driver_lib_dir/libcuda.so.1:$driver_lib_dir/libnvidia-ml.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
        else
            export LD_PRELOAD="$driver_lib_dir/libcuda.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
        fi
    elif [ -n "$real_nvml_lib" ]; then
        export LD_PRELOAD="$driver_lib_dir/libnvidia-ml.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    echo "b200_driver_libs real_libcuda=${real_libcuda:-<missing>} real_nvml_lib=${real_nvml_lib:-<missing>} ld_preload=${LD_PRELOAD:-<empty>}"
}

install_runtime_wheels() {
    local artifacts wheel_dir
    local specs=()
    local missing_wheels=()
    artifacts="$(artifact_dir)"
    wheel_dir="$artifacts/wheels"
    if [ ! -d "$wheel_dir" ]; then
        return 0
    fi

    if ! "$VIRTUAL_ENV/bin/python" -c 'import aiosqlite' >/dev/null 2>&1; then
        if compgen -G "$wheel_dir/aiosqlite-*.whl" >/dev/null; then
            specs+=("aiosqlite==0.22.1")
        else
            missing_wheels+=("aiosqlite")
        fi
    fi

    if ! "$VIRTUAL_ENV/bin/python" -c 'import mbridge' >/dev/null 2>&1; then
        if compgen -G "$wheel_dir/mbridge-*.whl" >/dev/null; then
            specs+=("mbridge==0.15.1")
        else
            missing_wheels+=("mbridge")
        fi
    fi

    if [ "${#missing_wheels[@]}" -gt 0 ]; then
        echo "Missing runtime wheel(s) in $wheel_dir: ${missing_wheels[*]}" >&2
        return 1
    fi

    if [ "${#specs[@]}" -gt 0 ]; then
        "$VIRTUAL_ENV/bin/python" -m pip install --no-index --no-deps --find-links "$wheel_dir" "${specs[@]}"
    fi

    "$VIRTUAL_ENV/bin/python" - <<'PY'
import aiosqlite
import mbridge

print(f"runtime_wheels_ok aiosqlite={aiosqlite.__file__} mbridge={mbridge.__file__}", flush=True)
PY
}

repair_verl_venv_python() {
    local pyvenv_cfg="$VIRTUAL_ENV/pyvenv.cfg"
    local pyhome=""
    local candidate=""

    if [ -f "$pyvenv_cfg" ]; then
        pyhome="$(awk -F' = ' '$1 == "home" {print $2}' "$pyvenv_cfg" | tail -1)"
    fi

    if [ -n "$pyhome" ] && [ -x "$pyhome/python3.12" ]; then
        candidate="$pyhome/python3.12"
    elif [ -n "$pyhome" ] && [ -x "$pyhome/python3" ]; then
        candidate="$pyhome/python3"
    elif [ -x /usr/bin/python3.12 ]; then
        candidate=/usr/bin/python3.12
    elif [ -x /usr/local/bin/python3.12 ]; then
        candidate=/usr/local/bin/python3.12
    fi

    if [ ! -x "$VIRTUAL_ENV/bin/python" ] && [ -n "$candidate" ]; then
        ln -sfn "$candidate" "$VIRTUAL_ENV/bin/python"
        ln -sfn python "$VIRTUAL_ENV/bin/python3"
        ln -sfn python "$VIRTUAL_ENV/bin/python3.12"
    fi

    if [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
        echo "Unable to restore executable Python for $VIRTUAL_ENV." >&2
        echo "pyvenv_cfg=$pyvenv_cfg pyhome=$pyhome" >&2
        ls -l "$VIRTUAL_ENV/bin/python"* >&2 || true
        exit 127
    fi
}

restore_verl_venv() {
    local artifacts venv_tar candidate
    artifacts="$(artifact_dir)"
    if [ ! -x /tmp/verl_venv/bin/python ]; then
        venv_tar="${RLLM_SWE_VERL_VENV_TAR:-}"
        if [ -z "$venv_tar" ]; then
            for candidate in \
                "$artifacts/verl_venv_920246.tar.gz" \
                /opt/tiger/swe_runtime_image/artifacts/verl_venv_920246.tar.gz \
                /mnt/hdfs/rllm_swe_artifacts/verl_venv_920246.tar.gz; do
                if [ -s "$candidate" ]; then
                    venv_tar="$candidate"
                    break
                fi
            done
        fi
        if [ -z "$venv_tar" ] || [ ! -s "$venv_tar" ]; then
            echo "Missing verl venv tar. RLLM_SWE_ARTIFACT_DIR=$artifacts RLLM_SWE_VERL_VENV_TAR=${RLLM_SWE_VERL_VENV_TAR:-<unset>}" >&2
            ls -lh "$artifacts" /opt/tiger/swe_runtime_image/artifacts /mnt/hdfs/rllm_swe_artifacts 2>&1 | sed 's/^/artifact_ls /' >&2 || true
            return 1
        fi
        echo "Restoring verl venv from $venv_tar"
        rm -rf /tmp/verl_venv
        tar -C /tmp -xzf "$venv_tar"
    fi
    export VIRTUAL_ENV=/tmp/verl_venv
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    repair_verl_venv_python
    install_runtime_wheels
}

restore_megatron_cp2_overlay() {
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

restore_qwen35_cache() {
    local artifacts
    if [ -n "${MODEL_PATH:-}" ] && [ -d "$MODEL_PATH" ]; then
        export HF_HOME=${HF_HOME:-/tmp/hf_cache}
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        return
    fi
    if [ -f /mnt/hdfs/model_path/config.json ]; then
        export MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/model_path}
        export HF_HOME=${HF_HOME:-/tmp/hf_cache}
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        return
    fi
    artifacts="$(artifact_dir)"
    if [ ! -d /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a ]; then
        test -s "$artifacts/qwen35_9b_hf_cache.tar"
        rm -rf /tmp/hf_cache/hub/models--Qwen--Qwen3.5-9B
        mkdir -p /tmp/hf_cache/hub
        tar -C /tmp/hf_cache/hub -xf "$artifacts/qwen35_9b_hf_cache.tar"
    fi
    export HF_HOME=/tmp/hf_cache
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
}

restore_rllm_home() {
    local artifacts rllm_home_tar
    artifacts="$(artifact_dir)"
    rllm_home_tar="${RLLM_SWE_RLLM_HOME_TAR:-}"
    if [ -z "$rllm_home_tar" ]; then
        if [ -s /mnt/hdfs/rllm_swe_artifacts/rllm_home_swe_data.tar.gz ]; then
            rllm_home_tar=/mnt/hdfs/rllm_swe_artifacts/rllm_home_swe_data.tar.gz
        else
            rllm_home_tar="$artifacts/rllm_home_swe_data.tar.gz"
        fi
    fi
    if [ ! -f /tmp/rllm_home/datasets/registry.json ]; then
        test -s "$rllm_home_tar"
        rm -rf /tmp/rllm_home /tmp/.rllm
        tar -C /tmp -xzf "$rllm_home_tar"
        mv /tmp/.rllm /tmp/rllm_home
    fi
    export RLLM_HOME=/tmp/rllm_home
}

restore_swe_artifacts() {
    echo "restore_swe_artifacts: setup_b200_driver_libs"
    setup_b200_driver_libs
    echo "restore_swe_artifacts: restore_verl_venv"
    restore_verl_venv
    echo "restore_swe_artifacts: restore_megatron_cp2_overlay"
    restore_megatron_cp2_overlay
    echo "restore_swe_artifacts: restore_qwen35_cache"
    restore_qwen35_cache
    echo "restore_swe_artifacts: restore_rllm_home"
    restore_rllm_home
    echo "restore_swe_artifacts: done"
}
