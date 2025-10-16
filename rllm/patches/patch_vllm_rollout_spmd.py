"""Patches for verl.workers.rollout.vllm_rollout.vllm_rollout_spmd module.

This module patches the vLLMAsyncRollout class to:
- Fix ZeroMQ socket initialization
- Add LoRA support with proper configuration
- Handle worker initialization with correct ranks
- Monkey-patch compute_logits after model loading
"""

from ._utils import vllm_max_lora_rank

TARGET_MODULE = "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd"


# ============================================================================
# Patches for vLLMAsyncRollout class
# ============================================================================


def _init_zeromq_wrapper(wrapped, instance, args, kwargs):
    """
    Replace _init_zeromq to fix socket initialization.

    Full replacement: does NOT call `wrapped`.
    """
    import getpass
    import os
    import threading

    import zmq
    from filelock import FileLock

    tensor_parallel_size = instance.config.tensor_model_parallel_size
    local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
    socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

    user = getpass.getuser()
    with FileLock(f"/tmp/verl_vllm_zmq_{user}.lock"):
        if socket_type == "ipc":
            pid = os.getpid()
            address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{user}.ipc"
        else:
            ip, port = instance._get_free_port()
            address = f"tcp://{ip}:{port}"

        context = zmq.Context()
        instance.socket = context.socket(zmq.REP)
        instance.socket.bind(address)

    instance.loop_thread = threading.Thread(target=instance._loop_forever)
    instance.loop_thread.start()
    return address


def _init_wrapper(wrapped, instance, args, kwargs):
    """
    Replace __init__ to add LoRA kwargs and defer engine initialization.

    Full replacement: does NOT call `wrapped`.
    Signature: (model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs)
    """
    # Extract parameters from args or kwargs (caller uses kwargs)
    if args:
        _, config, tokenizer, _ = args[:4]
    else:
        config = kwargs.get("config")
        tokenizer = kwargs.get("tokenizer")

    instance.tokenizer = tokenizer
    instance.config = config
    instance.inference_engine = None
    instance.sharding_manager = None
    instance.is_sleep = False
    instance.address = instance._init_zeromq()  # calls our wrapped version
    # Add LoRA config
    instance.lora_kwargs = kwargs.pop("lora_kwargs", {})
    # if you need to preserve any other kw, keep them on instance as well


def init_worker_wrapper(wrapped, instance, args, kwargs):
    """
    Replace init_worker to handle ranks and LoRA wiring.

    Full replacement: does NOT call `wrapped`.
    """
    import os

    from vllm.config import LoRAConfig
    from vllm.worker.worker_base import WorkerWrapperBase

    (all_kwargs,) = args  # expect list[dict[str, Any]]
    all_kwargs[0]["rank"] = int(os.environ["RANK"])
    all_kwargs[0]["local_rank"] = 0

    instance.vllm_config = all_kwargs[0]["vllm_config"]

    if getattr(instance, "lora_kwargs", None):
        lora_kwargs = {k: v for k, v in instance.lora_kwargs.items() if k != "enable_lora"}

        max_lora_rank = vllm_max_lora_rank(lora_kwargs["max_lora_rank"])
        lora_kwargs["max_lora_rank"] = max_lora_rank
        lora_config = LoRAConfig(**lora_kwargs)
        lora_config.verify_with_model_config(instance.vllm_config.model_config)
        instance.vllm_config.lora_config = lora_config

    instance.inference_engine = WorkerWrapperBase(vllm_config=instance.vllm_config)
    # ensure LoRA kwargs propagate to worker init
    instance.inference_engine.init_worker(all_kwargs)


def load_model_wrapper(wrapped, instance, args, kwargs):
    """
    Replace load_model to monkey-patch logits after worker load.

    Full replacement: calls engine's worker load_model rather than original.
    """
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _monkey_patch_compute_logits

    # delegate to the worker:
    instance.inference_engine.worker.load_model(*args, **kwargs)

    # sync sharding manager state
    instance.sharding_manager.inference_engine = instance.inference_engine
    instance.sharding_manager.model_runner = instance.inference_engine.worker.model_runner

    # patch compute_logits
    _monkey_patch_compute_logits(instance.inference_engine.worker.model_runner.model, len(instance.tokenizer))


# ============================================================================
# Registration
# ============================================================================


def register():
    """Register all patches for this module."""
    from ._utils import wrap_class_method_once

    cls_name = "vLLMAsyncRollout"
    wrap_class_method_once(TARGET_MODULE, cls_name, "_init_zeromq", _init_zeromq_wrapper)
    wrap_class_method_once(TARGET_MODULE, cls_name, "__init__", _init_wrapper)
    wrap_class_method_once(TARGET_MODULE, cls_name, "init_worker", init_worker_wrapper)
    wrap_class_method_once(TARGET_MODULE, cls_name, "load_model", load_model_wrapper)
