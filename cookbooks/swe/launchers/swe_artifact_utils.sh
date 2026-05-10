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
    local real_libcuda=/usr/lib/x86_64-linux-gnu/libcuda.so.1
    local real_nvml_lib=""
    local f
    for f in /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*; do
        if [ -f "$f" ] && [ -s "$f" ] && [ ! -L "$f" ]; then
            real_nvml_lib="$f"
            break
        fi
    done
    if [ -f "$real_libcuda" ] && [ -s "$real_libcuda" ]; then
        if [ -n "$real_nvml_lib" ]; then
            export LD_PRELOAD="$real_libcuda:$real_nvml_lib${LD_PRELOAD:+:$LD_PRELOAD}"
        else
            export LD_PRELOAD="$real_libcuda${LD_PRELOAD:+:$LD_PRELOAD}"
        fi
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
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
}

restore_qwen35_cache() {
    local artifacts
    if [ -n "${MODEL_PATH:-}" ] && [ -d "$MODEL_PATH" ]; then
        export HF_HOME=${HF_HOME:-/tmp/hf_cache}
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        return
    fi
    if [ -d /mnt/hdfs/model_path ]; then
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
