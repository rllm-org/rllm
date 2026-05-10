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
}

install_runtime_wheels() {
    local artifacts wheel_dir
    local specs=()
    artifacts="$(artifact_dir)"
    wheel_dir="$artifacts/wheels"
    if [ ! -d "$wheel_dir" ]; then
        return 0
    fi

    if compgen -G "$wheel_dir/aiosqlite-*.whl" >/dev/null; then
        specs+=("aiosqlite==0.22.1")
    fi
    if compgen -G "$wheel_dir/mbridge-*.whl" >/dev/null; then
        specs+=("mbridge==0.15.1")
    fi

    if [ "${#specs[@]}" -gt 0 ]; then
        python -m pip install --no-index --find-links "$wheel_dir" "${specs[@]}"
    fi
}

restore_verl_venv() {
    local artifacts
    artifacts="$(artifact_dir)"
    if [ ! -x /tmp/verl_venv/bin/python ]; then
        test -s "$artifacts/verl_venv_920246.tar.gz"
        rm -rf /tmp/verl_venv
        tar -C /tmp -xzf "$artifacts/verl_venv_920246.tar.gz"
    fi
    export VIRTUAL_ENV=/tmp/verl_venv
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    install_runtime_wheels
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
    local artifacts
    artifacts="$(artifact_dir)"
    if [ ! -f /tmp/rllm_home/datasets/registry.json ]; then
        test -s "$artifacts/rllm_home_swe_data.tar.gz"
        rm -rf /tmp/rllm_home /tmp/.rllm
        tar -C /tmp -xzf "$artifacts/rllm_home_swe_data.tar.gz"
        mv /tmp/.rllm /tmp/rllm_home
    fi
    export RLLM_HOME=/tmp/rllm_home
}

restore_swe_artifacts() {
    setup_b200_driver_libs
    restore_verl_venv
    restore_qwen35_cache
    restore_rllm_home
}
