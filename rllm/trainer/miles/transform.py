"""rLLM episodes -> Miles training samples.

Ported from ``rllm/trainer/tinker/transform.py``, minus the right-shift: Miles
consumes the raw ``tokens`` sequence and applies its own logits/token offset
internally, so we hand it flat tokens plus three response-aligned arrays.

A merged multi-turn sequence looks like ``[O1, A1, O2, A2, ...]``. Miles models a
sample as prompt + response with ``loss_mask`` over the response only, so the
prompt is ``O1`` and the response is everything after it, with the mask zeroed on
the intermediate observations.

The three response arrays must each be exactly ``response_length`` long:
``convert_samples_to_train_data`` asserts it for ``loss_mask``, and
``slice_log_prob_with_cp`` asserts it for the per-token float arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rllm.trainer.algorithms.step_merge import is_prefix, partition_steps_by_lineage
from rllm.types import Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


@dataclass
class SamplePayload:
    """One Miles ``Sample`` worth of data, without importing miles.

    Keeping this backend-SDK-free is what lets the token math be unit-tested off
    the Miles image.
    """

    tokens: list[int]
    prompt_length: int
    loss_mask: list[int]
    rollout_log_probs: list[float]
    advantages: list[float]
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def response_length(self) -> int:
        return len(self.tokens) - self.prompt_length

    def validate(self) -> None:
        n = self.response_length
        if n < 0:
            raise ValueError(f"prompt_length {self.prompt_length} exceeds token count {len(self.tokens)}")
        for name, arr in (
            ("loss_mask", self.loss_mask),
            ("rollout_log_probs", self.rollout_log_probs),
            ("advantages", self.advantages),
        ):
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected response_length {n}")


def _step_tokens(step, which: str) -> list[int]:
    ids = getattr(step, which) or []
    if not all(isinstance(t, int) for t in ids):
        raise TypeError(f"Miles backend needs flat integer {which}; got a chunked/multimodal token input. Multimodal rollouts are not supported on this backend yet.")
    return list(ids)


def _step_advantages(step, n_response: int) -> list[float]:
    if step.advantage is None:
        raise ValueError("step.advantage is None — compute advantages before transforming.")
    if isinstance(step.advantage, list):
        if len(step.advantage) != n_response:
            raise ValueError(f"per-token advantage has length {len(step.advantage)}, expected {n_response}")
        return [float(a) for a in step.advantage]
    return [float(step.advantage)] * n_response


@dataclass
class _Chain:
    """One merge chain being accumulated: a prompt, then interleaved action/observation tokens."""

    tokens: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    prompt_length: int = 0

    @property
    def trainable(self) -> bool:
        """False for a chain that never produced a response token (nothing to learn from)."""
        return self.prompt_length < len(self.tokens)

    def add_turn(self, delta: list[int], response: list[int], logprobs: list[float], advantages: list[float]) -> None:
        """Append an observation delta (masked out) followed by an action (masked in)."""
        pad = [0] * len(delta)
        self.tokens.extend(delta + response)
        self.mask.extend(pad + [1] * len(response))
        self.logprobs.extend([0.0] * len(delta) + logprobs)
        self.advantages.extend([0.0] * len(delta) + advantages)

    def to_payload(self, reward: float, metadata: dict) -> SamplePayload:
        cut = self.prompt_length
        payload = SamplePayload(
            tokens=list(self.tokens),
            prompt_length=cut,
            loss_mask=self.mask[cut:],
            rollout_log_probs=self.logprobs[cut:],
            advantages=self.advantages[cut:],
            reward=reward,
            metadata=metadata,
        )
        payload.validate()
        return payload


def trajectory_to_payloads(traj: Trajectory) -> list[SamplePayload]:
    """Merge a trajectory's steps into as few sequences as possible.

    Steps are partitioned by gateway lineage first, then each lineage is walked: a
    step whose prompt extends the accumulated sequence continues the chain, anything
    else closes it and starts a new one.
    """
    payloads: list[SamplePayload] = []
    reward = traj.reward or 0.0
    metadata = dict(traj.metadata or {})

    for lineage_steps in partition_steps_by_lineage(traj.steps):
        chain = _Chain()

        for step in lineage_steps:
            prompt = _step_tokens(step, "prompt_ids")
            response = _step_tokens(step, "response_ids")
            step_logprobs = [float(x) for x in (step.logprobs or [])]
            if len(step_logprobs) != len(response):
                raise ValueError(f"step has {len(response)} response tokens but {len(step_logprobs)} logprobs")
            step_advantages = _step_advantages(step, len(response))

            if not chain.tokens:
                delta = prompt
                chain.prompt_length = len(prompt)
            elif is_prefix(chain.tokens, prompt):
                delta = prompt[len(chain.tokens) :]
            else:
                if chain.trainable:
                    payloads.append(chain.to_payload(reward, metadata))
                chain = _Chain(prompt_length=len(prompt))
                delta = prompt

            chain.add_turn(delta, response, step_logprobs, step_advantages)

        if chain.trainable:
            payloads.append(chain.to_payload(reward, metadata))

    return payloads


def trajectory_groups_to_payloads(groups: list[TrajectoryGroup]) -> list[list[SamplePayload]]:
    """One inner list per group, so GRPO grouping survives into ``Sample.group_index``."""
    out: list[list[SamplePayload]] = []
    for group in groups:
        group_payloads: list[SamplePayload] = []
        for traj in group.trajectories:
            group_payloads.extend(trajectory_to_payloads(traj))
        if group_payloads:
            out.append(group_payloads)
        else:
            logger.warning("trajectory group %s produced no trainable samples; dropping", group.group_id)
    return out


def payloads_to_samples(grouped: list[list[SamplePayload]]) -> tuple[list, list[list[float]]]:
    """Build Miles ``Sample`` objects plus the parallel per-token advantage arrays.

    Advantages are returned separately rather than stuffed into ``Sample.metadata``:
    they have to land as a top-level ``rollout_data["advantages"]`` key so Miles'
    ``get_rollout_data`` CP-slices them alongside ``rollout_log_probs``. The two
    lists are index-aligned, and ``convert_samples_to_train_data`` preserves sample
    order, so the caller can attach them after conversion.

    ``index`` / ``group_index`` are set because ``convert_samples_to_train_data``
    still uses them for grouping and metrics even though we supply advantages.

    Imports miles, so this needs the Miles image.
    """
    from miles.utils.types import Sample

    samples: list[Sample] = []
    advantages: list[list[float]] = []
    index = 0
    for group_index, group in enumerate(grouped):
        for payload in group:
            payload.validate()
            samples.append(
                Sample(
                    group_index=group_index,
                    index=index,
                    tokens=payload.tokens,
                    response_length=payload.response_length,
                    loss_mask=payload.loss_mask,
                    rollout_log_probs=payload.rollout_log_probs,
                    reward=payload.reward,
                    status=Sample.Status.COMPLETED,
                    metadata=dict(payload.metadata),
                )
            )
            advantages.append(payload.advantages)
            index += 1
    return samples, advantages
