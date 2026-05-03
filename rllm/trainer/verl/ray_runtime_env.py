import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
]


def _get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = str(os.environ.get("RLLM_EXCLUDE")).split(",")
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    exclude_vars = set()
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            forward_prefix.remove(name.replace("*", "_"))
        else:
            exclude_vars.add(name)

    forwarded = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in forward_prefix) and k not in exclude_vars}
    return forwarded


def get_ppo_ray_runtime_env():
    """Build the runtime_env to pass to ray.init().

    Priority (low → high):
      1. PPO_RAY_RUNTIME_ENV — rllm defaults
      2. forwarded host env vars (VLLM_*, NCCL_*, CUDA_*, etc.)
      3. RAY_JOB_CONFIG_JSON_ENV_VAR — runtime_env from `ray job submit --runtime-env-json=...`

    Ray's ray.init() will merge the runtime_env we return here with the job config's
    runtime_env, and raises on any key conflict unless RAY_OVERRIDE_JOB_RUNTIME_ENV=1.
    We avoid that by popping any key the job config sets from our returned dict, so
    the job config's value wins.
    """
    env = PPO_RAY_RUNTIME_ENV.get("env_vars", {}).copy()
    env.update(_get_forwarded_env_vars())

    # Parse the job-submission runtime_env (if launched via `ray job submit`)
    try:
        job_runtime_env = json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}) or {}
    except (json.JSONDecodeError, TypeError):
        job_runtime_env = {}

    # Pop keys that the job config sets — let the job config's values win during ray.init merge
    for key in job_runtime_env.get("env_vars", {}) or {}:
        env.pop(key, None)

    runtime_env = {"env_vars": env}
    # Only set working_dir=None when the job config doesn't specify one (avoid merge conflict)
    if job_runtime_env.get("working_dir") is None:
        runtime_env["working_dir"] = None
    # Apply rLLM's verl patches (PR #5881 backport, dynamic-batch sync, etc.) on every
    # Ray worker process so the patches take effect inside FSDP workers — driver-side
    # monkey-patches do not propagate. The hook function is lazy and idempotent.
    if job_runtime_env.get("worker_process_setup_hook") is None:
        # Ray expects a dotted import path (no colon); it does
        # ``module.rpartition('.') -> module + attr`` to load the hook.
        runtime_env["worker_process_setup_hook"] = "rllm.experimental.verl.patch.apply_all_verl_patches"
    return runtime_env
