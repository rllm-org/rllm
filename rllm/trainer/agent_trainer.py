from collections.abc import Callable
from typing import Any, Literal

from rllm.data import Dataset


class AgentTrainer:
    """Wrapper that runs PPO training over a Workflow or an AgentFlow rollout function.

    Two ways to plug in your agent:

    * ``workflow_class`` — a :class:`rllm.workflows.workflow.Workflow` subclass
      (driven by :class:`rllm.engine.agent_workflow_engine.AgentWorkflowEngine`).
    * ``agent_run_func`` — a plain rollout function (driven by
      :class:`rllm.engine.agent_sdk_engine.AgentSdkEngine`).

    Backends:

    * ``verl`` (default): distributed PPO via the verl framework.
    * ``fireworks``: pipeline-based variant for the Fireworks workflow API.
    * ``tinker``: single-machine LoRA training via tinker.

    The legacy ``agent_class`` + ``env_class`` path that was driven by
    ``AgentExecutionEngine`` has been removed. New agents should be authored
    as a Workflow or as an AgentFlow rollout function (see ``cookbooks/``).
    """

    def __init__(
        self,
        workflow_class: type | None = None,
        workflow_args: dict[str, Any] | None = None,
        config: dict[str, Any] | list[str] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        backend: Literal["verl", "fireworks", "tinker"] = "verl",
        agent_run_func: Callable | None = None,
    ):
        """Initialize the AgentTrainer.

        Args:
            workflow_class: The workflow class to use for training.
            workflow_args: Optional arguments to pass to the workflow class.
            config: Configuration overrides to apply to the default config.
                Can be a dictionary with dot notation keys
                (e.g. ``{"data.train_batch_size": 8}``) or a list of strings
                (e.g. ``["data.train_batch_size=8"]``).
            train_dataset: Optional train dataset.
            val_dataset: Optional validation dataset.
            backend: Training backend (``'verl'`` | ``'fireworks'`` | ``'tinker'``).
            agent_run_func: Plain rollout function for the AgentSdk path.
        """
        assert backend in ("verl", "fireworks", "tinker"), f"Unsupported backend: {backend}; must be one of ('verl', 'fireworks', 'tinker')"
        self.backend = backend

        if backend == "fireworks" and workflow_class is None:
            raise ValueError("The 'fireworks' backend requires workflow_class.")

        if workflow_class is None and agent_run_func is None:
            raise ValueError("AgentTrainer requires either workflow_class or agent_run_func. The legacy agent_class + env_class interface is no longer supported.")

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.agent_run_func = agent_run_func

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        if self.backend == "verl":
            self._train_verl()
        elif self.backend == "fireworks":
            self._train_fireworks()
        elif self.backend == "tinker":
            self._train_tinker()

    def _train_tinker(self):
        if self.workflow_class is None:
            raise ValueError("The tinker backend requires workflow_class.")
        from rllm.trainer.deprecated.tinker_workflow_trainer import TinkerWorkflowTrainer

        trainer = TinkerWorkflowTrainer(
            config=self.config,
            workflow_class=self.workflow_class,
            workflow_args=self.workflow_args,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
        )
        trainer.fit_agent()

    def _train_verl(self):
        """Train using the standard verl backend."""
        import ray

        from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
        from rllm.trainer.verl.train_agent_ppo import TaskRunner

        if not ray.is_initialized():
            from rllm.trainer.ray_init_utils import get_ray_init_settings

            ray_init_settings = get_ray_init_settings(self.config)
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        runner_cls = ray.remote(num_cpus=1)(TaskRunner)
        runner = runner_cls.remote()

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                agent_run_func=self.agent_run_func,
            )
        )

    def _train_fireworks(self):
        """Train using the fireworks (pipeline) backend — workflow-only."""
        import ray

        if not ray.is_initialized():
            # TODO: check whether we need a separate function to retrieve the runtime environment (for fireworks)
            from verl.trainer.constants_ppo import get_ppo_ray_runtime_env as get_fireworks_ray_runtime_env

            ray.init(runtime_env=get_fireworks_ray_runtime_env(), num_cpus=self.config.ray_init.num_cpus)

        # Lazy import to avoid requiring fireworks package for users who don't use it
        from rllm.trainer.verl.train_workflow_pipeline import PipelineTaskRunner

        runner = PipelineTaskRunner.remote()

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
            )
        )
