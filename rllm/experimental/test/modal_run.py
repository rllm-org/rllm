"""
Modal launcher for rLLM + SkyRL training.

Builds a Modal image based on the SkyRL base image, mounts the local rLLM repo,
installs rLLM and its SkyRL dependencies, then runs a given command.

Usage:
    # Basic run (2x A100, solver-judge with Qwen3-0.6B)
    modal run rllm/experimental/test/modal_run.py

    # Custom command
    modal run rllm/experimental/test/modal_run.py --command "nvidia-smi"

    # With secrets from local shell env or repo-root .env
    modal run rllm/experimental/test/modal_run.py

    # Override GPU config
    MODAL_GPU=A100:4 modal run rllm/experimental/test/modal_run.py
"""

import os
from pathlib import Path

import modal


def _find_rllm_root() -> Path:
    """Find the rLLM repository root by walking up from this file.

    Returns:
        Path to the rLLM repo root (contains pyproject.toml and rllm/ directory).

    Raises:
        RuntimeError: If the repo root cannot be found.
    """
    if "RLLM_ROOT" in os.environ:
        return Path(os.environ["RLLM_ROOT"])

    start = Path(__file__).resolve()
    for base in [start] + list(start.parents):
        if (base / "pyproject.toml").exists() and (base / "rllm").is_dir():
            return base

    raise RuntimeError("rLLM repo root not found. Set RLLM_ROOT env var or run from within the repo.")


def _load_local_env_values(repo_root: Path, keys: tuple[str, ...]) -> dict[str, str]:
    """Load selected env vars from the shell first, then fall back to repo-root .env."""
    values = {key: value for key in keys if (value := os.environ.get(key))}
    env_path = repo_root / ".env"
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key not in keys or key in values:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            values[key] = value

    return values


def create_modal_secrets(repo_root: Path) -> list[modal.Secret]:
    """Create Modal-managed secrets for local API keys."""
    secret_values = _load_local_env_values(repo_root, ("RLLM_API_KEY", "RLLM_UI_URL", "WANDB_API_KEY"))
    if not secret_values:
        return []
    return [modal.Secret.from_dict(secret_values)]


def create_modal_image() -> modal.Image:
    """Creates a Modal image with rLLM + SkyRL installed.

    Uses the SkyRL base image (has Ray, vLLM, flash-attn, torch pre-installed),
    mounts the local rLLM repo, and installs rLLM + SkyRL packages.
    """
    local_rllm_root = _find_rllm_root()
    print(f"rLLM root: {local_rllm_root}")

    envs = {
        "RLLM_ROOT": "/root/rllm",
        "HF_HOME": "/root/data/hf_cache",
        "HYDRA_FULL_ERROR": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S": "100000000000",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }

    image = (
        modal.Image.from_registry("novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8")
        .env(envs)
        .add_local_dir(
            local_path=str(local_rllm_root),
            remote_path="/root/rllm",
            copy=True,
            ignore=[
                ".venv",
                ".env",
                ".env.*",
                "*.pyc",
                "__pycache__",
                ".git",
                "*.egg-info",
                ".pytest_cache",
                "node_modules",
                ".DS_Store",
            ],
        )
        .run_commands(
            # Install core CUDA stack needed by skyrl-train (torch -> flash-attn -> vllm),
            # then install local editable packages and pin transformers for vLLM compat.
            'pip install "torch==2.8.0" "torchvision==0.23.0"',
            'pip install "flash-attn==2.8.3" --no-build-isolation',
            'pip install "vllm==0.11.0"',
            "cd /root/rllm && pip install -e ./skyrl/skyrl-gym && pip install -e ./skyrl/skyrl-train && pip install -e ./skyrl && pip install -e .",
            'pip install "transformers==4.57.6"',
        )
    )

    return image


def create_modal_volume(volume_name: str = "rllm-data") -> dict[str, modal.Volume]:
    """Creates a persistent volume for checkpoints and HF cache.

    Args:
        volume_name: Name of the Modal volume. Creates it if it doesn't exist.

    Returns:
        Dict mapping mount path to Volume object.
    """
    data_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    return {"/root/data": data_volume}


def _infer_num_gpus(gpu_spec: str) -> int:
    """Infer GPU count from a Modal gpu spec like A100:2 or L4:1."""
    _, _, count = gpu_spec.partition(":")
    if count.isdigit():
        return int(count)
    return 1


app = modal.App(os.getenv("MODAL_APP_NAME", "rllm_skyrl_app_2"))
image = create_modal_image()
volume = create_modal_volume()
secrets = create_modal_secrets(_find_rllm_root())


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "A100:2"),
    volumes=volume,
    secrets=secrets,
    timeout=int(os.environ.get("MODAL_TIMEOUT", "28800")),
)
def run_script(command: str):
    """Run a command inside the rLLM container on Modal.

    Starts a Ray head node, then executes the given command from the rLLM repo root.
    Output is streamed live.
    """
    import subprocess

    repo_root = os.environ.get("RLLM_ROOT", "/root/rllm")

    print(f"Container repo root: {repo_root}")
    print(f"Working directory: {os.getcwd()}")

    os.chdir(repo_root)
    print(f"Changed to: {os.getcwd()}")

    # Start Ray head node
    print("Starting Ray head node...")
    subprocess.run("ray start --head", shell=True, check=True)
    os.environ["RAY_ADDRESS"] = "auto"
    os.environ.setdefault("NUM_GPUS", str(_infer_num_gpus(os.environ.get("MODAL_GPU", "A100:2"))))

    print(f"Running command: {command}")
    print("=" * 60)

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        print(line, end="")

    returncode = process.wait()

    print("=" * 60)
    if returncode != 0:
        raise Exception(f"Command failed with exit code {returncode}")


@app.local_entrypoint()
def main(command: str = "bash rllm/experimental/test/run_skyrl_solver_judge_modal.sh"):
    """Run rLLM + SkyRL training on Modal.

    Args:
        command: Command to run inside the container. Defaults to the solver-judge training script.

    Examples:
        modal run rllm/experimental/test/modal_run.py
        modal run rllm/experimental/test/modal_run.py --command "nvidia-smi"
        MODAL_GPU=A100:4 modal run rllm/experimental/test/modal_run.py --command "bash rllm/experimental/test/run_skyrl_solver_judge_modal.sh"
    """
    print(f"{'=' * 5} Submitting command to Modal: {command} {'=' * 5}")
    run_script.remote(command)
    print(f"\n{'=' * 5} Command completed successfully {'=' * 5}")
