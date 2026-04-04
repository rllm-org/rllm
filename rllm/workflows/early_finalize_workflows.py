from __future__ import annotations

from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.cumulative_workflow import CumulativeWorkflow
from rllm.workflows.early_finalize import attach_model_output_to_step, coerce_early_finalize_config, maybe_generate_with_early_finalize
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.single_turn_workflow import SingleTurnWorkflow


class EarlyFinalizeWorkflowMixin:
    def __init__(self, *args, early_finalize_config: Any = None, **kwargs):
        self.early_finalize_config = coerce_early_finalize_config(early_finalize_config, default_enable=True)
        super().__init__(*args, **kwargs)

    async def _generate_model_step(self, messages: list[dict], *, task: dict, application_id: str, **kwargs) -> tuple[ModelOutput, list[float] | None, dict | None]:
        generation = await maybe_generate_with_early_finalize(
            self,
            messages,
            application_id=application_id,
            task=task,
            **kwargs,
        )
        return generation.output, generation.response_mask, generation.metadata

    def _attach_model_step(
        self,
        current_step,
        output: ModelOutput,
        response_mask: list[float] | None,
        metadata: dict | None,
    ) -> None:
        attach_model_output_to_step(current_step, output, response_mask)
        if current_step is not None and metadata is not None:
            current_step.info["early_finalize"] = metadata


class SingleTurnWorkflowWithEarlyFinalize(EarlyFinalizeWorkflowMixin, SingleTurnWorkflow):
    pass


class MultiTurnWorkflowWithEarlyFinalize(EarlyFinalizeWorkflowMixin, MultiTurnWorkflow):
    pass


class CumulativeWorkflowWithEarlyFinalize(EarlyFinalizeWorkflowMixin, CumulativeWorkflow):
    pass
