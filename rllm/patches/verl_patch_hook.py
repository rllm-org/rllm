import sys

from wrapt import wrap_function_wrapper
from wrapt.importer import when_imported

_TARGETS = {
    "vllm_rollout_spmd": "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    "fsdp_vllm": "verl.workers.sharding_manager.fsdp_vllm",
}


def setup():
    _patch_vllm_rollout_spmd(_TARGETS["vllm_rollout_spmd"])
    _patch_fsdp_vllm(_TARGETS["fsdp_vllm"])


# helper: idempotent patch application
def _wrap_once(module_name: str, cls_name: str, method_name: str, wrapper):
    """Wrap a class method once, descriptor-safe, even if hot-reloaded/imported twice."""
    sentinel = f"__patched_{cls_name}_{method_name}__"
    assert method_name in ["_init_zeromq", "__init__", "init_worker", "load_model", "update_params"], f"{method_name} is not a class method"

    def apply(mod):
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls, sentinel, False):
            return
        wrap_function_wrapper(cls, method_name, wrapper)
        setattr(cls, sentinel, True)

    @when_imported(module_name)
    def _on_import(mod):
        apply(mod)

    # If already imported, patch immediately
    if module_name in sys.modules:
        apply(sys.modules[module_name])


# verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMAsyncRollout
def _patch_vllm_rollout_spmd(target_module: str):
    cls_name = "vLLMAsyncRollout"

    # _init_zeromq replacement
    def _init_zeromq_wrapper(wrapped, instance, args, kwargs):
        # full replacement: do NOT call `wrapped`
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

    # __init__ replacement (adds lora_kwargs, defers engine)
    def _init_wrapper(wrapped, instance, args, kwargs):
        # signature: (model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs)
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

    # init_worker replacement (handles ranks & LoRA wiring)
    def init_worker_wrapper(wrapped, instance, args, kwargs):
        # full replacement
        import os

        from vllm.config import LoRAConfig
        from vllm.worker.worker_base import WorkerWrapperBase

        (all_kwargs,) = args  # expect list[dict[str, Any]]
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        instance.vllm_config = all_kwargs[0]["vllm_config"]

        if getattr(instance, "lora_kwargs", None):
            lora_kwargs = {k: v for k, v in instance.lora_kwargs.items() if k != "enable_lora"}
            # Fix: for vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
            # verl mistakenly set `max_lora_rank = config.lora_rank`,this prevents us from using very small lora_rank (e.g. 1).
            vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
            lora_rank = lora_kwargs["max_lora_rank"]

            max_lora_idx = 0
            while max_lora_idx < len(vllm_max_lora_ranks) and vllm_max_lora_ranks[max_lora_idx] < lora_rank:
                max_lora_idx += 1

            lora_kwargs["max_lora_rank"] = vllm_max_lora_ranks[max_lora_idx]
            lora_config = LoRAConfig(**lora_kwargs)
            lora_config.verify_with_model_config(instance.vllm_config.model_config)
            instance.vllm_config.lora_config = lora_config

        instance.inference_engine = WorkerWrapperBase(vllm_config=instance.vllm_config)
        # ensure LoRA kwargs propagate to worker init
        instance.inference_engine.init_worker(all_kwargs)

    # load_model replacement (monkey-patch logits after worker load)
    def load_model_wrapper(wrapped, instance, args, kwargs):
        # full replacement: call engine's worker load_model rather than original
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _monkey_patch_compute_logits

        # delegate to the worker:
        instance.inference_engine.worker.load_model(*args, **kwargs)

        # sync sharding manager state
        instance.sharding_manager.inference_engine = instance.inference_engine
        instance.sharding_manager.model_runner = instance.inference_engine.worker.model_runner

        # patch compute_logits
        _monkey_patch_compute_logits(instance.inference_engine.worker.model_runner.model, len(instance.tokenizer))

    # Install wrappers (idempotent)
    _wrap_once(target_module, cls_name, "_init_zeromq", _init_zeromq_wrapper)
    _wrap_once(target_module, cls_name, "__init__", _init_wrapper)
    _wrap_once(target_module, cls_name, "init_worker", init_worker_wrapper)
    _wrap_once(target_module, cls_name, "load_model", load_model_wrapper)


# verl.workers.sharding_manager.fsdp_vllm.FSDPVLLMShardingManager
def _patch_fsdp_vllm(target_module: str):
    cls_name = "FSDPVLLMShardingManager"

    def update_params_wrapper(wrapped, instance, args, kwargs):
        # full replacement
        import importlib as _importlib
        import logging
        import os
        from dataclasses import asdict

        from verl.utils.vllm_utils import TensorLoRARequest, patch_vllm_moe_model_weight_loader

        try:
            # torch 2.5+
            from torch.distributed.tensor import DTensor
        except ImportError:
            from torch.distributed._tensor import DTensor  # noqa: F401

        updated_params = kwargs.get("updated_params") if "updated_params" in kwargs else args[0]
        peft_config = kwargs.get("peft_config") if "peft_config" in kwargs else (args[1] if len(args) > 1 else None)

        # Use module-level logger if present, else fallback by module name
        mod = _importlib.import_module(instance.__class__.__module__)
        logger = getattr(mod, "logger", None) or logging.getLogger(mod.__name__)
        if not getattr(logger, "level", None):
            logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

        model = instance.model_runner.model
        if peft_config and instance.base_sync_done:
            from verl.utils.vllm_utils import TensorLoRARequest  # local import for clarity

            lora_int_id = 123
            ie = instance.inference_engine
            if hasattr(ie, "llm_engine"):
                ie.llm_engine.remove_lora(lora_int_id)
            else:
                ie.worker.remove_lora(lora_int_id)

            lora_request = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(updated_params),
            )
            # async mode (WorkerWrapperBase): prefer llm_engine if present
            if hasattr(ie, "llm_engine"):
                ie.llm_engine.add_lora(lora_request)
            else:
                ie.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
            return
        else:
            patch_vllm_moe_model_weight_loader(model)

            loaded_params = model.load_weights(updated_params)
            logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")

    _wrap_once(target_module, cls_name, "update_params", update_params_wrapper)
