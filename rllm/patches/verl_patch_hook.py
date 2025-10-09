import wrapt
from typing import Any

_TARGETS = {
    "vllm_rollout_spmd": "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    "fsdp_vllm": "verl.workers.sharding_manager.fsdp_vllm",
}


def setup():
    _patch_vllm_rollout_spmd(_TARGETS["vllm_rollout_spmd"])
    _patch_fsdp_vllm(_TARGETS["fsdp_vllm"])


def _patch_vllm_rollout_spmd(target):
    @wrapt.when_imported(target)
    def _patch(mod):
        Cls = getattr(mod, "vLLMAsyncRollout", None)
        if Cls is None:
            return
            
        # Patch 1: _init_zeromq
        def _patched_init_zeromq(self) -> str:
            import getpass
            import os
            import threading
            import zmq
            from filelock import FileLock

            tensor_parallel_size = self.config.tensor_model_parallel_size
            local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
            socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

            user = getpass.getuser()
            with FileLock(f"/tmp/verl_vllm_zmq_{user}.lock"):
                if socket_type == "ipc":
                    pid = os.getpid()
                    address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{user}.ipc"
                else:
                    ip, port = self._get_free_port()
                    address = f"tcp://{ip}:{port}"

                context = zmq.Context()
                self.socket = context.socket(zmq.REP)
                self.socket.bind(address)

            self.loop_thread = threading.Thread(target=self._loop_forever)
            self.loop_thread.start()
            return address

        # Patch 2: LoRA related
        original_init = Cls.__init__
        def _patched_init(self, *args, **kwargs):
            self.lora_kwargs = kwargs.pop("lora_kwargs", {})
            original_init(self, *args, **kwargs)

        def _patched_init_worker(self, all_kwargs: list[dict[str, Any]]):
            from vllm.config import LoRAConfig
            from vllm.worker.worker_base import WorkerWrapperBase
            import os
            
            all_kwargs[0]["rank"] = int(os.environ["RANK"])
            all_kwargs[0]["local_rank"] = 0

            self.vllm_config = all_kwargs[0]["vllm_config"]

            if self.lora_kwargs:
                lora_kwargs = {k: v for k, v in self.lora_kwargs.items() if k != "enable_lora"}
                lora_config = LoRAConfig(**lora_kwargs)
                model_config = self.vllm_config.model_config
                lora_config.verify_with_model_config(model_config)
                self.vllm_config.lora_config = lora_config

            self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
            # Fix: now if self.lora_kwargs is not empty, the lora kwargs are also passed to the init_worker
            self.inference_engine.init_worker(all_kwargs)

        def _patched_load_model(self, *args, **kwargs):
            from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _monkey_patch_compute_logits
            # Fix: the `load_model` is a function of the worker
            self.inference_engine.worker.load_model(*args, **kwargs)

            # inference engine is initialized now, update sharding manager
            self.sharding_manager.inference_engine = self.inference_engine
            self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

            _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

        Cls._init_zeromq = _patched_init_zeromq
        Cls.__init__ = _patched_init
        Cls._init_worker = _patched_init_worker
        Cls.load_model = _patched_load_model


def _patch_fsdp_vllm(target):
    @wrapt.when_imported(target)
    def _patch(mod):
        Cls = getattr(mod, "FSDPVLLMShardingManager", None)
        if Cls is None:
            return

        def _patched_update_params(self, updated_params, peft_config=None):
            import os
            import time
            import logging
            from dataclasses import asdict
            from verl.utils.vllm_utils import TensorLoRARequest, patch_vllm_moe_model_weight_loader
            from verl.utils.model import check_exclude_modules, check_target_modules
            from verl.utils.device import get_device_id
            
            try:
                # for torch 2.5+
                from torch.distributed.tensor import DTensor
            except ImportError:
                from torch.distributed._tensor import DTensor

            # Use the module's logger if available (best for consistency)
            logger = getattr(mod, 'logger', None)
            if logger is None:
                # Fallback: use module name to match original verl logging hierarchy
                logger = logging.getLogger(mod.__name__)
                logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

            model = self.model_runner.model
            if peft_config:
                if self.base_sync_done:
                    lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                    lora_request = TensorLoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="simon_lora_path",
                        peft_config=asdict(peft_config),
                        lora_tensors=updated_params,
                    )
                    # Fix: for async mode (WorkerWrapperBase), use add_lora() directly
                    if hasattr(self.inference_engine, 'llm_engine'):
                        self.inference_engine.llm_engine.add_lora(lora_request)
                    else:
                        self.inference_engine.add_lora(lora_request)
                    logger.info(f"vLLM load weights, loaded_params: {len(updated_params)}")
                    return
                else:
                    def replace_lora_wrapper(k):
                        """Replace LoRA parameter keys with base layer equivalents.

                        Transforms LoRA parameter names to their corresponding base layer
                        names for proper weight loading in vLLM when base model sync is not done.

                        Args:
                            k (str): Original parameter key name.

                        Returns:
                            str: Transformed parameter key for base layer.
                        """
                        stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                        if k.endswith(".weight"):
                            module_k = k[: -len(".weight")]
                            if check_exclude_modules(peft_config, module_k):
                                return k
                            elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(
                                peft_config, module_k
                            ):
                                return f"{module_k}.base_layer.weight"
                        if k.endswith(".bias"):
                            module_k = k[: -len(".bias")]
                            if check_exclude_modules(peft_config, module_k):
                                return k
                            elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(
                                peft_config, module_k
                            ):
                                return f"{module_k}.base_layer.bias"
                        return k

                    updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

            patch_vllm_moe_model_weight_loader(model)
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            loaded_params = model.load_weights(
                (
                    (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                    for name, param in updated_params.items()
                )
            )

            self.base_sync_done = True
            logger.info(f"vLLM load weights, loaded_params: {len(loaded_params) if loaded_params else -1}")

        Cls.update_params = _patched_update_params