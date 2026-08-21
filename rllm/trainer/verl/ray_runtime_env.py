import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from rllm.trainer.ray_init_utils import FORWARD_PREFIXES, get_forwarded_env_vars  # noqa: F401  (re-exported)

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
    env.update(get_forwarded_env_vars())

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
    return runtime_env
