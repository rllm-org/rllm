from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout_engine import RolloutEngine
from rllm.workflows.workflow import TerminationReason, Workflow


@dataclass
class StrandsEvent:
    """
    Minimal unified event type for Strands streaming.
    """

    type: str  # "TextDelta" | "ToolUse" | "ToolResult" | "Stop"
    text: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    toolUseId: Optional[str] = None
    content: Optional[Any] = None
    status: Optional[str] = None  # "success" | "error"
    final_text: Optional[str] = None
    logprob: Optional[float] = None


class StrandsWorkflow(Workflow):
    """
    Skeleton Workflow that will consume a Strands Agent event stream and
    package it into an rLLM Episode/Trajectory/Step structure.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        strands_session_factory,  # callable: (system_prompt, tools, **kw) -> session with .step(user_msg) -> AsyncIterator[StrandsEvent]
        system_prompt: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 8,
        reward_fn=None,
        **kwargs,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, **kwargs)
        self.session_factory = strands_session_factory
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_steps = max_steps
        self.reward_fn = reward_fn

    async def __call__(self, task: dict | str, uid: str, **kwargs) -> Episode:
        """
        Minimal placeholder that returns a single-step Episode with the
        system and user messages wired in. Full event-stream handling to be
        implemented.
        """
        # Normalize task to a string message for the initial chat
        task_text = task if isinstance(task, str) else str(task)

        chat: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_text},
        ]

        # Minimal single-turn generation so users can see a reply
        try:
            assistant_text = await self.rollout_engine.get_model_response(chat)
            if isinstance(assistant_text, str) and assistant_text:
                chat.append({"role": "assistant", "content": assistant_text})
        except Exception:
            # Keep silent on generation failures in skeleton
            pass

        trajectory = Trajectory(steps=[], reward=0.0)
        trajectory.steps.append(Step(chat_completions=list(chat)))

        # Optionally compute a terminal reward via reward_fn
        if self.reward_fn is not None:
            try:
                trajectory.reward = float(
                    self.reward_fn(trajectory=trajectory, final_text="", steps=len(trajectory.steps))
                )
            except Exception:
                trajectory.reward = 0.0

        episode = Episode(
            id=uid,
            task=task,
            trajectories=[("solver", trajectory)],
            is_correct=None,
            metrics={"num_steps": len(trajectory.steps)},
        )
        episode.termination_reason = TerminationReason.ENV_DONE
        return episode


