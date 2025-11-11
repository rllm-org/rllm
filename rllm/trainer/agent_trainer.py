from typing import Any

import ray

from rllm.data import Dataset


class AgentTrainer:
    """
    A wrapper class that allows users to easily train custom agents with custom environments
    without having to directly interact with the underlying training infrastructure.
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
        backend: str = "verl",
    ):
        """
        Initialize the AgentTrainer.

        Args:
            agent_class: The custom agent class to use for training
            env_class: The custom environment class to use for training
            config: Configuration overrides to apply to the default config
                   Can be a dictionary with dot notation keys (e.g., {"data.train_batch_size": 8})
                   or a list of strings in the format "key=value" (e.g., ["data.train_batch_size=8"])
            train_dataset: Optional train dataset to use
            val_dataset: Optional validation dataset to use
            agent_args: Optional arguments to pass to the agent class
            env_args: Optional arguments to pass to the environment class
        """
        if workflow_class is not None:
            if agent_class is not None:
                raise ValueError(
                    "agent_class is not supported when using workflow, instead use workflow_args['agent_cls']"
                )
            if agent_args is not None:
                raise ValueError(
                    "agent_args is not supported when using workflow, instead use workflow_args['agent_args']"
                )
            if env_class is not None:
                raise ValueError(
                    "env_class is not supported when using workflow, instead use workflow_args['env_cls']"
                )
            if env_args is not None:
                raise ValueError(
                    "env_args is not supported when using workflow, instead use workflow_args['env_args']"
                )

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}

        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.backend = backend

        assert self.backend in [
            "verl", "tinker"
        ], f"Unsupported backend: {self.backend}, must be one of ['verl', 'tinker']"

        if train_dataset is not None and self.config is not None and hasattr(
                self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(
                self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        if self.backend == "verl":
            self._train_verl()
        elif self.backend == "tinker":
            self._train_tinker()

    def _train_tinker(self):
        from rllm.trainer.tinker.tinker_agent_trainer import TinkerAgentTrainer
        from rllm.trainer.tinker.tinker_workflow_trainer import TinkerWorkflowTrainer

        if self.workflow_class is not None:
            trainer = TinkerWorkflowTrainer(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
            )
        else:
            trainer = TinkerAgentTrainer(
                config=self.config,
                agent_class=self.agent_class,
                env_class=self.env_class,
                agent_args=self.agent_args,
                env_args=self.env_args,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
            )
        trainer.fit_agent()

    def _train_verl(self):
        from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
        from rllm.trainer.verl.train_agent_ppo import TaskRunner
        # Check if Ray is not initialized
        if not ray.is_initialized():
            # read off all the `ray_init` settings from the config
            if self.config is not None and hasattr(self.config, "ray_init"):
                ray_init_settings = {
                    k: v
                    for k, v in self.config.ray_init.items() if v is not None
                }
            else:
                ray_init_settings = {}
            ray.init(runtime_env=get_ppo_ray_runtime_env(),
                     **ray_init_settings)

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
            ))
