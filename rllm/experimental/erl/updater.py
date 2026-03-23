"""ErlPromptUpdater — queries the policy model for an improved system prompt."""

from __future__ import annotations

from typing import Any

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.experimental.erl.utils import UPDATER_SYSTEM_PROMPT, extract_prompt_from_response


class ErlPromptUpdater:
    """Query the shared policy model for an improved system prompt.

    The updater sends contextual state (recent attempts, feedback, metrics)
    to the rollout engine and extracts a revised prompt from the model's
    response.  It uses the **same** rollout engine as the solver, so no
    extra model is needed.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        system_prompt: str = UPDATER_SYSTEM_PROMPT,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        self.rollout_engine = rollout_engine
        self.system_prompt = system_prompt
        self.sampling_params = sampling_params or {"temperature": 0.7, "top_p": 0.9}

    async def propose_prompt(
        self,
        state: str,
        current_prompt: str,
    ) -> tuple[str, Trajectory]:
        """Ask the model for a revised prompt.

        Args:
            state: Contextual information (attempts, feedback, metrics).
            current_prompt: The prompt to improve.

        Returns:
            A ``(new_prompt, updater_trajectory)`` tuple.  If extraction
            fails, *new_prompt* falls back to *current_prompt*.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": state},
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, **self.sampling_params)
        content = output.content or output.text or ""
        extracted = extract_prompt_from_response(content)
        new_prompt = extracted if extracted else current_prompt

        step = Step(
            chat_completions=messages + [{"role": "assistant", "content": content}],
            thought=output.reasoning or "",
            action=new_prompt,
            model_output=output,
            metadata={"previous_prompt": current_prompt},
        )
        trajectory = Trajectory(
            name="erl_updater",
            steps=[step],
            metadata={"previous_prompt": current_prompt},
        )
        return new_prompt, trajectory
