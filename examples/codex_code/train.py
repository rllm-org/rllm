import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.harnesses.codex import CodexHarness
from rllm.trainer import AgentTrainer

from examples.sandbox_code.prepare_data import prepare_sandbox_code_data
from examples.sandbox_code.sandbox_code_eval import sandbox_code_evaluator


class PreinstalledCodexHarness(CodexHarness):
    """CodexHarness with install_script disabled + nvm removed from invocation.

    Codex is pre-installed on host via apt (nodesource) and visible inside
    bwrap via /usr read-only bind-mount. No nvm needed.
    """

    def install_script(self) -> str:
        return ""

    def build_invocation(self, instruction, task, config):
        inv = super().build_invocation(instruction, task, config)
        return inv.replace(". $HOME/.nvm/nvm.sh 2>/dev/null; ", "")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    prepare_sandbox_code_data()
    train_dataset = DatasetRegistry.load_dataset("sandbox_code", "train")
    test_dataset = DatasetRegistry.load_dataset("sandbox_code", "test")

    sandbox_backend = config.get("sandbox_backend", "bwrap")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=PreinstalledCodexHarness(),
        evaluator=sandbox_code_evaluator,
        sandbox_backend=sandbox_backend,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
