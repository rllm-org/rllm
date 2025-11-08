from typing import Any, Literal

import ray

from rllm.data import Dataset
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
from rllm.trainer.verl.train_agent_ppo import TaskRunner
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env as get_fireworks_ray_runtime_env


class AgentTrainer:
    """
    A wrapper class that allows users to easily train custom agents with custom environments
    without having to directly interact with the underlying training infrastructure.

    Supports two backends:
    - 'verl' (default): Standard training backend supporting both workflow and agent/env classes
    - 'fireworks': Pipeline-based training backend optimized for workflow-based training
    """

    def __init__(
        self,
        workflow_class: type | None = None,
        workflow_args: dict[str, Any] | None = None,
        agent_class: type | None = None,
        env_class: type | None = None,
        agent_args: dict[str, Any] | None = None,
        env_args: dict[str, Any] | None = None,
        config: dict[str, Any] | list[str] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        backend: Literal["verl", "fireworks"] = "verl",
    ):
        """
        Initialize the AgentTrainer.

        Args:
            workflow_class: The workflow class to use for training
            workflow_args: Optional arguments to pass to the workflow class
            agent_class: The custom agent class to use for training
            env_class: The custom environment class to use for training
            agent_args: Optional arguments to pass to the agent class
            env_args: Optional arguments to pass to the environment class
            config: Configuration overrides to apply to the default config
                   Can be a dictionary with dot notation keys (e.g., {"data.train_batch_size": 8})
                   or a list of strings in the format "key=value" (e.g., ["data.train_batch_size=8"])
            train_dataset: Optional train dataset to use
            val_dataset: Optional validation dataset to use
            backend: Training backend to use ('verl' or 'fireworks'). Default is 'verl'
        """
        # Validate backend
        if backend not in ["verl", "fireworks"]:
            raise ValueError(f"backend must be either 'verl' or 'fireworks', got '{backend}'")

        self.backend = backend

        # Validate backend-specific requirements
        if backend == "fireworks":
            if agent_class is not None or env_class is not None:
                raise ValueError("The 'fireworks' backend only supports workflow_class. agent_class and env_class are not supported. Use workflow_args to configure agent and environment.")
            if agent_args is not None or env_args is not None:
                raise ValueError("The 'fireworks' backend does not support agent_args or env_args. Use workflow_args to configure the workflow.")

        if workflow_class is not None and config is not None and hasattr(config, "rllm") and hasattr(config.rllm, "workflow") and config.rllm.workflow.use_workflow:
            if agent_class is not None:
                raise ValueError("agent_class is not supported when using workflow, instead use workflow_args['agent_cls']")
            if agent_args is not None:
                raise ValueError("agent_args is not supported when using workflow, instead use workflow_args['agent_args']")
            if env_class is not None:
                raise ValueError("env_class is not supported when using workflow, instead use workflow_args['env_cls']")
            if env_args is not None:
                raise ValueError("env_args is not supported when using workflow, instead use workflow_args['env_args']")

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}

        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.config = config

        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        """
        Start the training process using the specified backend.
        """
        if self.backend == "verl":
            self._train_with_verl()
        elif self.backend == "fireworks":
            self._train_with_fireworks()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _train_with_verl(self):
        """
        Train using the standard verl backend.
        Supports both workflow-based and agent/env-based training.
        """
        # Check if Ray is not initialized
        if not ray.is_initialized():
            # read off all the `ray_init` settings from the config
            if self.config is not None and hasattr(self.config, "ray_init"):
                ray_init_settings = {k: v for k, v in self.config.ray_init.items() if v is not None}
            else:
                ray_init_settings = {}
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        runner = TaskRunner.remote()

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                agent_class=self.agent_class,
                env_class=self.env_class,
                agent_args=self.agent_args,
                env_args=self.env_args,
            )
        )

    def _train_with_fireworks(self):
        """
        Train using the fireworks (pipeline) backend.
        Optimized for workflow-based training with the Fireworks API.
        """
        if not ray.is_initialized():
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
