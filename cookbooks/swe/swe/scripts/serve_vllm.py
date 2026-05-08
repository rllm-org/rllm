"""Serve Qwen3.5-35B-A3B-FP8 with OpenAI-compatible vLLM on Modal.

Requires vllm>=0.17.0 for qwen3_5_moe architecture support.

Deploy:
    export MODAL_APP_NAME="vllm-qwen35"
    modal deploy serve_vllm.py

Warm up:
    curl -L -sS --connect-timeout 10 --max-time 300 \
    https://rllm-project--vllm-qwen35-serve.modal.run/v1/models

Use with run_n_eval.py:
    export HOSTED_VLLM_API_BASE="https://rllm-project--vllm-qwen35-serve.modal.run/v1"
    export HOSTED_VLLM_API_KEY="fake-api-key"
    python swe/scripts/run_n_eval.py \
        --dataset swe_smith_py \
        --model "hosted_vllm/Qwen/Qwen3.5-35B-A3B-FP8" \
        --n_runs 4 --n_parallel 192 \
        --output_dir results/qwen3.5/

Host locally (requires vllm>=0.17.0 installed):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --host 0.0.0.0 --port 8000 \
        --data-parallel-size 8 --tensor-parallel-size 1 \
        --max-model-len 32768 --enable-prefix-caching \
        --enable-auto-tool-choice --tool-call-parser qwen3_xml




"""

import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

MINUTES = 60
N_GPU = 4
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B-FP8")
VLLM_PORT = 8000
MAX_MODEL_LEN = 32768
VLLM_NIGHTLY_SPEC = "vllm>=0.17.0"

# Cold starts are much faster with eager mode; use False for best steady-state throughput.
FAST_BOOT = True

# Optional: pin a HF commit for reproducibility.
MODEL_REVISION = os.environ.get("MODEL_REVISION") or None

app = modal.App(os.environ.get("MODAL_APP_NAME", "vllm-qwen35"))

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .dockerfile_commands(
        "COPY --from=ghcr.io/astral-sh/uv:latest /uv /.uv/uv",
    )
    .run_commands(
        "/.uv/uv pip install --python $(command -v python) --compile-bytecode "
        f"'{VLLM_NIGHTLY_SPEC}'",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=30 * MINUTES,
    timeout=24 * 60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=VLLM_PORT, startup_timeout=45 * MINUTES)
def serve():
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--uvicorn-log-level=info",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
        "--data-parallel-size", str(N_GPU),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--served-model-name", MODEL_NAME,
        "--enable-prefix-caching",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
    ]
    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    print("Launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    health_url = f"http://127.0.0.1:{VLLM_PORT}/health"
    deadline = time.time() + (45 * MINUTES)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {proc.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    print(f"vLLM is ready at {health_url}")
                    return
        except urllib.error.URLError:
            pass
        time.sleep(2)

    proc.terminate()
    raise TimeoutError(f"Timed out waiting for vLLM readiness at {health_url}")