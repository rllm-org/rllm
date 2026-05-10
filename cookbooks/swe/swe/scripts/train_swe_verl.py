#!/usr/bin/env python3
"""Train SWE with rLLM AgentTrainer using verl + FSDP + vLLM.

Uses the AgentFlow/Evaluator framework: SWEAgentFlow runs the agent,
SWEEvaluator grades the patch, and rllm's AgentTrainer handles the
GRPO training loop with gateway-mediated trace capture.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_BASE_DIR))

load_dotenv(_BASE_DIR / ".env")

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer

from swe.agent_flow import SWEAgentFlow
from swe.evaluator import SWEEvaluator
from swe.flow_config import SWEAgentFlowConfig


def _startup_log(message: str):
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[startup {timestamp}] {message}", flush=True)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {value!r}")


def _maybe_pin_task_runner_to_head():
    """Patch the verl launcher placement when requested.

    If RLLM_RUN_TASK_RUNNER_LOCAL=1, run the workflow runner in this Python
    process. This is useful when the driver is on a CPU node and the Ray
    cluster is only the GPU workers; SWE/Modal traffic then stays on the CPU
    driver while model workers are scheduled through Ray.

    If RLLM_PIN_TASK_RUNNER_TO_HEAD=1, schedule the TaskRunner on the same Ray
    node as the driver process.

    This keeps SWEAgentFlow / Modal connections on the head machine while
    GPU work runs on remote Ray workers.
    """
    run_task_runner_local = _env_flag("RLLM_RUN_TASK_RUNNER_LOCAL")
    pin_task_runner_to_head = _env_flag("RLLM_PIN_TASK_RUNNER_TO_HEAD")
    if not run_task_runner_local and not pin_task_runner_to_head:
        return

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from rllm.experimental.verl import verl_launcher
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    from rllm.trainer.ray_init_utils import get_ray_init_settings

    def _init_ray(config):
        if ray.is_initialized():
            _startup_log("ray already initialized")
            return

        ray_address = os.environ.get("RAY_ADDRESS", "")
        if run_task_runner_local and ray_address and "://" not in ray_address and ray_address != "auto":
            host = ray_address.rsplit(":", 1)[0]
            os.environ["RAY_ADDRESS"] = f"ray://{host}:10001"
            _startup_log(
                "RLLM_RUN_TASK_RUNNER_LOCAL=1: using Ray Client address "
                f"{os.environ['RAY_ADDRESS']} for external CPU driver"
            )

        ray_init_settings = get_ray_init_settings(config)
        if run_task_runner_local and os.environ.get("RAY_ADDRESS", "").startswith("ray://"):
            ray_init_settings["address"] = os.environ["RAY_ADDRESS"]

        runtime_env = get_ppo_ray_runtime_env()
        # Propagate credentials to all Ray workers (remote GPU nodes).
        env_vars = runtime_env.setdefault("env_vars", {})
        for key in (
            "LD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_REQUIRE_CUDA",
            "NVIDIA_DRIVER_CAPABILITIES",
        ):
            env_vars.pop(key, None)
        env_vars["RLLM_PATCH_TORCH_COALESCING_MANAGER"] = "1"
        for key in (
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "OPENAI_API_KEY",
            "WANDB_API_KEY",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "HYDRA_FULL_ERROR",
            "NCCL_DEBUG",
            "PYTHONFAULTHANDLER",
            "PYTHONUNBUFFERED",
            "RAY_BACKEND_LOG_LEVEL",
            "RAY_DEDUP_LOGS",
            "RAY_LOG_TO_DRIVER",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "VERL_LOGGING_LEVEL",
            "VLLM_LOGGING_LEVEL",
            "XDG_CACHE_HOME",
        ):
            val = os.environ.get(key)
            if val:
                env_vars[key] = val
        _startup_log(
            "ray.init starting "
            f"address={ray_init_settings.get('address', os.environ.get('RAY_ADDRESS', 'auto'))} "
            f"runtime_env_env_vars={sorted(env_vars)}"
        )
        ray.init(runtime_env=runtime_env, **ray_init_settings)
        _startup_log(f"ray.init done resources={ray.available_resources()}")

    if run_task_runner_local:
        from rllm.trainer.verl.train_agent_ppo import TaskRunner

        class LocalWorkflowTaskRunner(TaskRunner):
            def run(self, config, workflow_class, workflow_args, **kwargs):  # type: ignore[override]
                import os
                import socket
                from pprint import pprint

                import ray
                from omegaconf import OmegaConf
                from verl.trainer.ppo.reward import load_reward_manager
                from verl.trainer.ppo.utils import need_critic, need_reference_policy
                import verl.single_controller.ray.base as verl_ray_base
                from verl.single_controller.ray.base import ResourcePoolManager
                from verl.utils import hf_processor, hf_tokenizer
                from verl.utils.config import validate_config
                from verl.utils.fs import copy_to_local

                from rllm.experimental.unified_trainer import UnifiedTrainer
                from rllm.experimental.verl.verl_backend import VerlBackend

                _startup_log(f"LocalWorkflowTaskRunner hostname={socket.gethostname()} pid={os.getpid()}")
                pprint(OmegaConf.to_container(config))
                OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
                OmegaConf.resolve(config)
                _startup_log("hydra config resolved")

                legacy_mode = config.trainer.get("use_legacy_worker_impl", "auto")
                if legacy_mode != "disable":
                    config.trainer.use_legacy_worker_impl = "disable"

                _startup_log("adding verl worker roles")
                actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
                self.add_critic_worker(config)
                self.add_ref_policy_worker(config, actor_rollout_cls)

                _startup_log("validating verl config")
                validate_config(
                    config=config,
                    use_reference_policy=need_reference_policy(config),
                    use_critic=need_critic(config),
                )

                _startup_log(f"copy_to_local model path={config.actor_rollout_ref.model.path}")
                local_path = copy_to_local(
                    config.actor_rollout_ref.model.path,
                    use_shm=config.actor_rollout_ref.model.get("use_shm", False),
                )
                _startup_log(f"copy_to_local done local_path={local_path}")
                trust_remote_code = config.data.get("trust_remote_code", False)
                _startup_log("loading tokenizer")
                tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
                _startup_log("loading processor")
                processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
                if processor is not None and getattr(processor, "chat_template", None) is None:
                    processor = None
                    _startup_log("processor has no chat_template; using tokenizer path")

                def _check_resource_available_with_ray_client(self):
                    cluster_resources = ray.available_resources()
                    total_available_gpus = cluster_resources.get("GPU", cluster_resources.get("NPU", 0))
                    total_required_gpus = sum(
                        n_gpus
                        for process_on_nodes in self.resource_pool_spec.values()
                        for n_gpus in process_on_nodes
                    )
                    if total_available_gpus < total_required_gpus:
                        raise ValueError(
                            f"Total available GPUs {total_available_gpus} is less than "
                            f"total desired GPUs {total_required_gpus}"
                        )

                def _sort_placement_groups_without_private_state(pgs):
                    return pgs

                worker_env = {
                    key: val
                    for key in ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME")
                    if (val := os.environ.get(key))
                }
                if worker_env:
                    original_ray_worker_group_init = verl_ray_base.RayWorkerGroup.__init__

                    def _ray_worker_group_init_with_cache_env(self, *args, **kwargs):
                        merged_worker_env = dict(kwargs.get("worker_env") or {})
                        merged_worker_env.update(worker_env)
                        kwargs["worker_env"] = merged_worker_env
                        original_ray_worker_group_init(self, *args, **kwargs)

                    verl_ray_base.RayWorkerGroup.__init__ = _ray_worker_group_init_with_cache_env

                # Ray Client does not support Verl's private state API calls from
                # the external CPU driver. Use public aggregate resources and keep
                # placement groups in creation order.
                ResourcePoolManager._check_resource_available = _check_resource_available_with_ray_client
                verl_ray_base.sort_placement_group_by_node_ip = _sort_placement_groups_without_private_state

                _startup_log("loading reward managers")
                reward_fn = load_reward_manager(
                    config,
                    tokenizer,
                    **config.reward_model.get("reward_kwargs", {}),
                )
                val_reward_fn = load_reward_manager(
                    config,
                    tokenizer,
                    **config.reward_model.get("reward_kwargs", {}),
                )
                _startup_log("initializing resource pool manager")
                resource_pool_manager = self.init_resource_pool_mgr(config)

                backend_args = {
                    "tokenizer": tokenizer,
                    "processor": processor,
                    "role_worker_mapping": self.role_worker_mapping,
                    "resource_pool_manager": resource_pool_manager,
                    "ray_worker_group_cls": ray_worker_group_cls,
                    "reward_fn": reward_fn,
                    "val_reward_fn": val_reward_fn,
                }

                trainer = None
                try:
                    _startup_log("constructing UnifiedTrainer / VerlBackend")
                    trainer = UnifiedTrainer(
                        backend_cls=VerlBackend,
                        config=config,
                        workflow_class=workflow_class,
                        train_dataset=None,
                        val_dataset=None,
                        workflow_args=workflow_args,
                        backend_args=backend_args,
                        **kwargs,
                    )
                    _startup_log("UnifiedTrainer constructed; starting trainer.fit")
                    trainer.fit()
                finally:
                    if trainer is not None:
                        _startup_log("shutting down trainer")
                        trainer.shutdown()

        def _local_train(self):
            _startup_log("local trainer launcher starting")
            _init_ray(self.config)
            runner = LocalWorkflowTaskRunner()
            _startup_log("running LocalWorkflowTaskRunner")
            runner.run(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                store=self.store,
                **self.kwargs,
            )

        verl_launcher.VerlTrainerLauncher.train = _local_train
        return

    if not pin_task_runner_to_head:
        return

    def _pinned_train(self):
        _startup_log("pinned trainer launcher starting")
        _init_ray(self.config)

        head_node_id = ray.get_runtime_context().get_node_id()
        runner = verl_launcher.WorkflowTaskRunner.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head_node_id,
                soft=False,
            ),
        ).remote()

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                store=self.store,
                **self.kwargs,
            )
        )

    verl_launcher.VerlTrainerLauncher.train = _pinned_train


_maybe_pin_task_runner_to_head()


def _load_registered_dataset(dataset_name: str):
    dataset = DatasetRegistry.load_dataset(dataset_name)
    if dataset is None:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' is not registered. "
            f"Run: python -m swe.prepare_rllm_data --dataset {dataset_name}"
        )

    verl_path = dataset.get_verl_data_path()
    if verl_path is None:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' is missing its _verl.parquet companion. "
            f"Run: python -m swe.prepare_rllm_data --dataset {dataset_name}"
        )

    return dataset, verl_path


def _maybe_limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError(f"Dataset limit must be positive, got {limit}")
    return dataset.select(range(min(limit, len(dataset))))


def _maybe_limit_verl_parquet(dataset_name: str, verl_path: str, limit: int | None, split_name: str) -> str:
    if limit is None:
        return verl_path
    if limit <= 0:
        raise ValueError(f"Dataset limit must be positive, got {limit}")

    import polars as pl

    subset_dir = Path(HydraConfig.get().run.dir) / "verl_subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_path = subset_dir / f"{dataset_name}_{split_name}_{limit}_verl.parquet"

    if not subset_path.exists():
        pl.scan_parquet(verl_path).head(limit).collect().write_parquet(subset_path)

    return str(subset_path.resolve())


@hydra.main(version_base=None, config_path="../config", config_name="verl_swe_trainer")
def main(config: DictConfig):
    train_limit = config.get("train_max_samples")
    val_limit = config.get("val_max_samples")
    train_dataset_name = config.get("train_dataset", "swe_smith")
    val_dataset_name = config.get("val_dataset", None)

    train_dataset, train_verl_path = _load_registered_dataset(train_dataset_name)
    train_dataset = _maybe_limit_dataset(train_dataset, train_limit)
    train_verl_path = _maybe_limit_verl_parquet(train_dataset_name, train_verl_path, train_limit, "train")
    config.data.train_files = train_verl_path

    if val_dataset_name:
        val_dataset, val_verl_path = _load_registered_dataset(val_dataset_name)
        val_dataset = _maybe_limit_dataset(val_dataset, val_limit)
        val_verl_path = _maybe_limit_verl_parquet(val_dataset_name, val_verl_path, val_limit, "val")
        config.data.val_files = val_verl_path
    else:
        val_dataset = None
        # verl requires val_files to be a valid path even when unused;
        # point it at the train parquet so dataset loading doesn't crash.
        config.data.val_files = train_verl_path

    model_name = config.actor_rollout_ref.model.path
    swe_config = OmegaConf.to_container(config.get("swe", {}), resolve=True)

    # Mirror verl's rollout sampling into the agent's chat.completions kwargs so
    # the agent and verl's vLLM rollout sample identically. setdefault preserves
    # any explicit `+swe.model_temperature=...` override from the launch script.
    rollout_cfg = config.actor_rollout_ref.rollout
    if "temperature" in rollout_cfg:
        swe_config.setdefault("model_temperature", float(rollout_cfg.temperature))
    if "top_p" in rollout_cfg:
        swe_config.setdefault("model_top_p", float(rollout_cfg.top_p))

    print("=" * 60)
    print("SWE verl Training (AgentFlow + AgentTrainer)")
    print("=" * 60)
    print(f"Train dataset: {train_dataset_name} ({len(train_dataset)} instances)")
    print(f"Train verl path: {train_verl_path}")
    if val_dataset is not None:
        print(f"Val dataset: {val_dataset_name} ({len(val_dataset)} instances)")
        print(f"Val verl path: {config.data.val_files}")
    else:
        print("Val dataset: none")
    print(f"Model: {model_name}")
    print(f"Parallel tasks: {config.rllm.workflow.n_parallel_tasks}")
    print(f"Rollout n: {config.actor_rollout_ref.rollout.n}")
    print(f"Checkpoint dir: {config.trainer.default_local_dir}")
    print("=" * 60)

    # SWEAgentFlowConfig.from_config() accepts flat keys (swe.step_limit=150,
    # swe.val_step_limit=200, swe.compaction_enabled=true, ...) and nested
    # sections (swe.validation.step_limit=200). Unknown keys are dropped.
    _startup_log("building SWE flow config")
    flow_config = SWEAgentFlowConfig.from_config(swe_config)
    _startup_log("constructing SWEAgentFlow")
    agent_flow = SWEAgentFlow(flow_config)
    _startup_log("constructing SWEEvaluator")
    evaluator = SWEEvaluator(
        command_timeout=swe_config.get("command_timeout", 120),
        sandbox_timeout=swe_config.get("sandbox_timeout", 3600),
        verbose=swe_config.get("verbose", False),
    )

    _startup_log("constructing AgentTrainer")
    trainer = AgentTrainer(
        config=config,
        agent_flow=agent_flow,
        evaluator=evaluator,
        train_dataset=None,
        val_dataset=None,
        backend="verl",
    )
    _startup_log("starting trainer.train")
    trainer.train()


if __name__ == "__main__":
    main()
