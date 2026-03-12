import os
import subprocess

import modal

app = modal.App("skyrl-tx")

# Persistent volumes
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("skyrl-checkpoints", create_if_missing=True)

# Change this to the model you want to serve
BASE_MODEL = "Qwen/Qwen3-0.6B"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "libnuma-dev")
    .pip_install("uv")
    .run_commands("git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL.git /opt/skyrl")
    .run_commands("cd /opt/skyrl && uv sync --python $(which python3) --extra tinker --extra fsdp --no-dev")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .workdir("/opt/skyrl")
)


@app.function(
    image=image,
    gpu="A100:2",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/checkpoints": checkpoints,
    },
    timeout=86400,  # 24h max container lifetime
    scaledown_window=1800,  # 30min idle before shutdown
    max_containers=1,  # Single container to keep session state consistent
)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    # Remove stale database from previous runs to avoid "Deleting models not yet implemented" crash
    db_path = "/opt/skyrl/skyrl/tinker/tinker.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    subprocess.Popen(
        [
            "/opt/skyrl/.venv/bin/python",
            "-m",
            "skyrl.tinker.api",
            "--base-model",
            BASE_MODEL,
            "--backend",
            "fsdp",
            "--port",
            "8000",
        ],
        cwd="/opt/skyrl",
    )
