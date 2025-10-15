"""Patches for verl.workers.sharding_manager.fsdp_vllm module.

This module patches the FSDPVLLMShardingManager class to:
- Fix update_params to properly handle LoRA weight updates
- Support dynamic LoRA adapter loading/unloading
"""

from ._utils import LORA_ID_MAGIC, LORA_PATH_MAGIC, get_logger, wrap_class_method_once

TARGET_MODULE = "verl.workers.sharding_manager.fsdp_vllm"


# ============================================================================
# Patches for FSDPVLLMShardingManager class
# ============================================================================
def replace_lora_wrapper(k, peft_config):
    """Replace LoRA parameter keys with base layer equivalents.

    Transforms LoRA parameter names to their corresponding base layer
    names for proper weight loading in vLLM when base model sync is not done.

    Args:
        k (str): Original parameter key name.

    Returns:
        str: Transformed parameter key for base layer.
    """
    from verl.utils.model import check_exclude_modules, check_target_modules

    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if k.endswith(".weight"):
        module_k = k[: -len(".weight")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.weight"
    if k.endswith(".bias"):
        module_k = k[: -len(".bias")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.bias"
    return k


def _enter_wrapper(wrapped, instance, args, kwargs):
    """Patch __enter__ to properly handle LoRA weight updates."""
    try:
        # torch 2.5+
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor  # noqa: F401

    import inspect
    from collections import OrderedDict

    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

    from verl.utils.device import get_device_id, get_device_name, get_torch_device
    from verl.utils.fsdp_utils import (
        load_fsdp_model_to_gpu,
        offload_fsdp_model_to_cpu,
    )
    from verl.utils.model import convert_weight_keys
    from verl.utils.profiler import log_gpu_memory_usage, simple_timer

    logger = get_logger(instance.__class__.__module__)

    def __collect_lora_params() -> OrderedDict:
        """
        collect lora params or full params if base model is not ready in vllm
        work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
        """
        from peft.utils.save_and_load import get_peft_model_state_dict

        from verl.utils.fsdp_utils import (
            fsdp_version,
            layered_summon_lora_params,
        )

        lora_params = OrderedDict()
        peft_model = getattr(instance.module, "_fsdp_wrapped_module", instance.module)
        if fsdp_version(instance.module) > 0:
            if instance.layered_summon:
                if not instance.base_sync_done:
                    raise ValueError("To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let rollout.load_format=safetensors")
                lora_params = layered_summon_lora_params(instance.module)
            else:
                with FSDP.summon_full_params(instance.module, writeback=False):
                    if instance.base_sync_done:
                        lora_params = get_peft_model_state_dict(peft_model)
                        lora_params = {name: param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu() for name, param in lora_params.items()}
                    else:
                        model = peft_model.base_model.model
                        orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                        model = model.to("cpu")
                        for name, param in model.state_dict().items():
                            if any(x in name for x in ["_flat_param", "lora_"]):
                                continue
                            name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                            lora_params[name] = param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
                        model = model.to(orig_dev)
                get_torch_device().empty_cache()
        else:
            if instance.base_sync_done:
                lora_params = get_peft_model_state_dict(peft_model)
            else:
                model = peft_model.base_model.model
                orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                model = model.to("cpu")
                for name, param in model.state_dict().items():
                    if any(x in name for x in ["_flat_param", "lora_"]):
                        continue
                    name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                    lora_params[name] = param.detach().cpu()
                model = model.to(orig_dev)
        return lora_params

    # NOTE: Basically, we only need `get_torch_device().empty_cache()` before vllm wake_up and
    # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
    # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
    # to speed up memory allocations.
    #
    # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
    # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
    instance.timing = {}
    with simple_timer("reshard", instance.timing):
        get_torch_device().empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if instance.offload_param:
            load_fsdp_model_to_gpu(instance.module)

        peft_config = None
        peft_model = getattr(instance.module, "_fsdp_wrapped_module", instance.module)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = __collect_lora_params()
            if not instance.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = instance.module.state_dict()
        params = convert_weight_keys(params, getattr(instance.module, "_fsdp_wrapped_module", instance.module))
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        # fix: support per_tensor_param
        if peft_config is not None and instance.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = ((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param) for name, param in params.items())

        if instance.rollout_config.free_cache_engine:
            if "tags" in inspect.signature(instance.inference_engine.wake_up).parameters:
                instance.inference_engine.wake_up(tags=["weights"])
            else:
                instance.inference_engine.wake_up()

        # update model params
        instance.update_params(per_tensor_param, peft_config=peft_config)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
        del params, per_tensor_param
        if instance.offload_param:
            offload_fsdp_model_to_cpu(instance.module)
        get_torch_device().empty_cache()

        if instance.rollout_config.free_cache_engine and "tags" in inspect.signature(instance.inference_engine.wake_up).parameters:
            instance.inference_engine.wake_up(tags=["kv_cache"])

        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)
        instance.base_sync_done = True

        # important: need to manually set the random states of each tp to be identical.
        if instance.device_mesh is not None:
            instance.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(instance.gen_random_states)


def update_params_wrapper(wrapped, instance, args, kwargs):
    """
    Replace update_params to properly handle LoRA weight updates.

    Full replacement: does NOT call `wrapped`.
    """
    from dataclasses import asdict

    from verl.utils.vllm_utils import TensorLoRARequest, patch_vllm_moe_model_weight_loader

    try:
        # torch 2.5+
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor  # noqa: F401

    updated_params = kwargs.get("updated_params") if "updated_params" in kwargs else args[0]
    peft_config = kwargs.get("peft_config") if "peft_config" in kwargs else (args[1] if len(args) > 1 else None)

    logger = get_logger(instance.__class__.__module__)

    model = instance.model_runner.model
    if peft_config and instance.base_sync_done:
        from verl.utils.vllm_utils import TensorLoRARequest

        lora_int_id = LORA_ID_MAGIC
        ie = instance.inference_engine
        if hasattr(ie, "llm_engine"):
            ie.llm_engine.remove_lora(lora_int_id)
        else:
            ie.worker.remove_lora(lora_int_id)

        lora_request = TensorLoRARequest(
            lora_name=f"{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path=LORA_PATH_MAGIC,
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


# ============================================================================
# Registration
# ============================================================================


def register():
    """Register all patches for this module."""

    cls_name = "FSDPVLLMShardingManager"
    wrap_class_method_once(TARGET_MODULE, cls_name, "__enter__", _enter_wrapper)
    wrap_class_method_once(TARGET_MODULE, cls_name, "update_params", update_params_wrapper)
