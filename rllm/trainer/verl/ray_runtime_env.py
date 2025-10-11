import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },
    "worker_process_setup_hook": "rllm.patches.verl_patch_hook.setup",
}


def get_ppo_ray_runtime_env():
    """
    Return the PPO Ray runtime environment.
    Avoid repeating env vars already set in the driver env.
    """
    env_vars = PPO_RAY_RUNTIME_ENV["env_vars"].copy()
    for key in list(env_vars.keys()):
        if os.environ.get(key) is not None:
            env_vars.pop(key, None)

    return {
        "env_vars": env_vars,
        "worker_process_setup_hook": PPO_RAY_RUNTIME_ENV["worker_process_setup_hook"],
    }
