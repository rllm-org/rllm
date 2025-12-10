# Installation Guide

This guide will help you set up rLLM on your system.

## Prerequisites

Starting with v0.2.1, rLLM's recommended dependency manager is `uv`. To install `uv`, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

rLLM requires `python>=3.10`, but certain beckends may require a newer installation (e.g., `tinker` requires `python>=3.11`). Ensure that your system has a suitable installation of Python:
 
```bash
uv python install 3.11
```

## Basic Installation

The following will perform a minimal installation of rLLM:

```bash
git clone https://github.com/rllm-org/rllm.git
cd rllm

uv venv --python 3.11
uv pip install -e .
```

rLLM supports multiple backends for training, including `verl` and `tinker`, which need to be installed separately.

To train with `tinker` on a CPU-only machine, run:

```bash
uv pip install -e .[tinker] --torch-backend=cpu
```

To train with `verl` on a GPU-equipped machine with CUDA 12.8, run:
```bash
uv pip install -e .[verl] --torch-backend=cu128
```

> The `verl` extra installs vLLM by default. If you'd rather use SGLang to sample rollouts, you can install it with `uv pip install sglang --torch-backend=cu128`.

> rLLM with verl supports alternative hardware accelerators, including AMD ROCm and Huawei Ascend. For these platforms, we strongly recommend installing rLLM on top of verl's official Docker containers for ROCm ([here](https://github.com/volcengine/verl/tree/main/docker/rocm)) and Ascend ([here](https://github.com/volcengine/verl/tree/main/docker/ascend)).

### Activating your environment

Be sure to activate the virtual environment before running a job:

```bash
source .venv/bin/activate
python your_script.py
```

### Editable Verl Installation

If you wish to make changes to verl, you can do an editable install:

```bash 
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.6.1
uv pip install -e .
```

### Optional Extras

rLLM provides additional optional dependencies for specific agent domains and framework integrations. For example:
- `web`: Tools for web agents (BrowserGym, Selenium).
- `code-tools`: Sandboxed code execution (E2B, Together).
- `smolagents`: Integration with Hugging Face's smolagents.

See the full list of managed extras [here](pyproject.toml).

## Installation without `uv`

While rLLM can also be installed without `uv` (i.e., just using `pip`), it is not recommended and may cause issues if you don't have a compatible PyTorch or CUDA version preinstalled:

```bash
conda create -n rllm python=3.11
conda activate rllm
pip install -e .[verl]
```

## Installation with Docker 🐳

For a containerized setup, you can use Docker:

```bash
# Build the Docker image
docker build -t rllm .

# Create and start the container
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
docker start rllm-container

# Enter the container
docker exec -it rllm-container bash
```

For more help, refer to the [GitHub issues page](https://github.com/rllm-org/rllm/issues). 