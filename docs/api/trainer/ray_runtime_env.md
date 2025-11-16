# Ray Runtime Environment Configuration

## Overview

The `ray_runtime_env` module automatically forwards relevant environment variables from your local environment to Ray worker processes during distributed training. This ensures that configuration for libraries like VLLM, NCCL, CUDA, and HuggingFace are properly propagated to all workers.

## Environment Variable Forwarding

### Automatic Forwarding

Environment variables with the following prefixes are automatically forwarded to Ray workers:

- **Inference Engines**: `VLLM_`, `SGL_`, `SGLANG_`
- **HuggingFace Libraries**: `HF_`, `TOKENIZERS_`, `DATASETS_`
- **Training Frameworks**: `TORCH_`, `PYTORCH_`, `DEEPSPEED_`, `MEGATRON_`
- **CUDA/NCCL**: `NCCL_`, `CUDA_`, `CUBLAS_`, `CUDNN_`, `NV_`, `NVIDIA_`

### Default Environment Variables

The following variables are set by default for PPO training:

```python
{
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "VLLM_LOGGING_LEVEL": "WARN",
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "VLLM_USE_V1": "1",
}
```

Environment variables from your shell **can override** these defaults.

## Controlling Forwarding with RLLM_EXCLUDE

Use the `RLLM_EXCLUDE` environment variable to prevent specific variables or entire prefixes from being forwarded to Ray workers.

### Exclude Specific Variables

Exclude individual environment variables by name:

```bash
export RLLM_EXCLUDE="CUDA_VISIBLE_DEVICES,HF_TOKEN"
# CUDA_VISIBLE_DEVICES and HF_TOKEN will NOT be forwarded
```

### Exclude Entire Prefixes

Use the wildcard pattern `PREFIX*` to exclude all variables with a given prefix:

```bash
export RLLM_EXCLUDE="VLLM*"
# All VLLM_* variables will NOT be forwarded (except defaults)
```

### Combined Exclusions

Combine multiple exclusions with commas:

```bash
export RLLM_EXCLUDE="VLLM*,CUDA*,NCCL_IB_DISABLE"
# Excludes all VLLM_*, all CUDA_*, and the specific NCCL_IB_DISABLE variable
```

## Usage Example

```python
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

# Get the runtime environment configuration
runtime_env = get_ppo_ray_runtime_env()

# Pass to Ray actor initialization
actor = ActorClass.options(runtime_env=runtime_env).remote()
```

## Common Use Cases

### Debugging with Verbose Logging

```bash
export VLLM_LOGGING_LEVEL="DEBUG"
export NCCL_DEBUG="INFO"
# These will override defaults and propagate to all workers
```

### Preventing Token Forwarding

```bash
export RLLM_EXCLUDE="HF_TOKEN"
# Useful if you want workers to use a different authentication method
```

## API Reference

::: rllm.trainer.verl.ray_runtime_env._get_forwarded_env_vars
    options:
      show_root_heading: true
      show_source: true

::: rllm.trainer.verl.ray_runtime_env.get_ppo_ray_runtime_env
    options:
      show_root_heading: true
      show_source: true

::: rllm.trainer.verl.ray_runtime_env.FORWARD_PREFIXES
    options:
      show_root_heading: true
      show_source: false

