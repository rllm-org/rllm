import logging

from torch.distributed.device_mesh import init_device_mesh

from rllm.agents.agent import Trajectory

logger = logging.getLogger(__name__)


class AgentSFTTrainer:
    def __init__(self, config, train_dataset=None, val_dataset=None, backend="verl"):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.backend = backend

        assert self.backend in ["verl", "tinker"], f"Unsupported backend: {self.backend}, must be one of ['verl', 'tinker']"

    def train(self):
        """Start training with the selected backend."""
        if self.backend == "verl":
            self._train_verl()
        elif self.backend == "tinker":
            self._train_tinker()

    def _train_verl(self):
        from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
        from verl.utils import hf_tokenizer
        from verl.utils.device import get_device_name
        from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
        from verl.utils.fs import copy_to_local

        from rllm.trainer.verl.sft_dataset import RLLMSFTDataset

        config = self.config
        device_name = get_device_name()
        local_rank, rank, world_size = initialize_global_process_group()

        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
            mesh_dim_names=("dp", "sp"),
        )
        # build tokenizer and datasets first
        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

        train_dataset = RLLMSFTDataset(config.data.train_files, tokenizer, config.data)
        val_dataset = RLLMSFTDataset(config.data.val_files, tokenizer, config.data)

        trainer = FSDPSFTTrainer(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.fit()

        destroy_global_process_group()

    def _train_tinker(self):
        """Train using Tinker backend."""
        from rllm.trainer.tinker.tinker_sft_trainer import TinkerSFTTrainer

        trainer = TinkerSFTTrainer(
            config=self.config,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
        )
        trainer.fit_sft()

    @staticmethod
    def process_trajectories(trajectories: list[Trajectory], reward_threshold: float):
        """Process trajectories into SFT format."""
        sft_data = []

        for traj in trajectories:
            if not traj:
                continue

            reward = traj.reward

            if reward < reward_threshold:
                continue

            # Get chat_completions from the last step of the trajectory
            messages = None
            if traj.steps and hasattr(traj.steps[-1], "chat_completions"):
                messages = traj.steps[-1].chat_completions

            if not messages:
                continue

            clean_messages = [{"role": msg["role"], "content": str(msg["content"]).strip()} for msg in messages if isinstance(msg, dict) and msg.get("role") and str(msg.get("content", "")).strip()]

            if len(clean_messages) >= 2:
                sft_data.append({"messages": clean_messages})

        print(f"Processed {len(trajectories)} trajectories -> {len(sft_data)} valid examples")
        return sft_data
