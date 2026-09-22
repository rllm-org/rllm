"""Tinker-compatible policy trainer for Fireworks serverless training."""

from __future__ import annotations

import json
import os
from typing import Any

import tinker

from rllm.trainer.tinker.tinker_policy_trainer import (
    TinkerPolicyTrainer,
    require_training_client,
)


class FireworksServerlessPolicyTrainer(TinkerPolicyTrainer):
    """Use Fireworks' service-owned sampler for each saved LoRA snapshot."""

    def __init__(self, *args: Any, tokenizer: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    @require_training_client
    def create_sampling_client(self, sampler_path: str) -> tinker.SamplingClient:
        return self.service_client.create_sampling_client(
            model_path=sampler_path,
            tokenizer=self.tokenizer,
        )

    @require_training_client
    async def save_checkpoint_and_get_sampling_client(
        self,
        batch_idx: int,
        do_save: bool = False,
        dataloader_state: dict | None = None,
    ) -> tinker.SamplingClient:
        # Fireworks serverless does not implement Tinker's
        # save_weights_and_get_sampling_client shortcut. A sampler must be
        # opened through the service against an explicit snapshot.
        name = f"{batch_idx:06d}"
        sampler_future = await self.training_client.save_weights_for_sampler_async(name)

        state_future = None
        if do_save:
            state_future = await self.training_client.save_state_async(name)

        sampler_path = (await sampler_future.result_async()).path
        if do_save:
            assert state_future is not None
            state_path = (await state_future.result_async()).path
            step_dir = os.path.join(
                self.config.training.default_local_dir,
                f"global_step_{batch_idx}",
            )
            os.makedirs(step_dir, exist_ok=True)
            with open(os.path.join(step_dir, "checkpoint.json"), "w") as f:
                json.dump(
                    {
                        "name": name,
                        "state_path": state_path,
                        "sampler_path": sampler_path,
                        "dataloader_state": dataloader_state,
                    },
                    f,
                )
            with open(
                os.path.join(
                    self.config.training.default_local_dir,
                    "latest_checkpointed_iteration.txt",
                ),
                "w",
            ) as f:
                f.write(str(batch_idx))

        return self.create_sampling_client(sampler_path)
