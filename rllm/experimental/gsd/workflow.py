"""GsdWorkflow — Generalized Self-Distillation workflow.

This is the active GSD implementation (see :mod:`rllm.experimental.gsd.legacy`
for the archived Top-K variant).  Key design choices:

1. **Direct sampling client access.** The workflow builds ``ModelInput``
   objects via the renderer's ``build_prompt`` method and calls
   ``sampling_client.sample_async`` directly, bypassing
   ``rollout_engine.get_model_response``.  This avoids the
   ``ModelOutput → Step.from_model_output`` round-trip and lets us use
   ``num_samples=N`` for single-call batched rollouts.

2. **Frozen reference teacher.** Teacher rollouts and teacher logprob
   evaluations go through a :class:`FrozenTeacherRef` that pins the
   initial sampling client at the start of training so its logprobs
   stay stable across ``optim_step`` calls.  Student rollouts and hint
   generation use the *live* (updating) sampling client.

3. **SFT-style CE.** The CE loss is classic cross-entropy on
   ``(student_prompt, teacher_response)`` pairs via Tinker's built-in
   ``cross_entropy`` loss.  No Top-K, no extra ``sample_async`` call.

4. **Sampled-token IS.** The IS loss uses per-token reverse-KL advantages
   ``teacher_lp - student_lp`` computed against the frozen teacher via
   a single ``compute_logprobs_async`` call per student response.

5. **Hint GRPO.** When ``train_hint=True``, the hint generation step
   becomes a trajectory with reward ``R_T_avg - R_S_avg``.  The custom
   grouping hook (see :mod:`rllm.experimental.gsd.grouping`) merges
   hint trajectories across tasks into a single group so REINFORCE has
   a meaningful baseline.

Trajectory roles produced:

* :data:`rllm.experimental.gsd.losses.CE_ROLE` (``"gsd_ce"``)
* :data:`rllm.experimental.gsd.losses.IS_ROLE` (``"gsd_is"``)
* :data:`rllm.experimental.gsd.losses.GRPO_ROLE` (``"gsd_grpo"``) — Case 2
* :data:`rllm.experimental.gsd.losses.HINT_ROLE` (``"gsd_hint"``) — when train_hint=True
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import tinker

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.experimental.gsd.losses import (
    CE_ROLE,
    GRPO_ROLE,
    HINT_ROLE,
    IS_ROLE,
    compute_teacher_logprobs_for_response,
    kl_advantages_from_logprobs,
)
from rllm.experimental.gsd.prompts import (
    build_hint_prompt,
    build_student_prompt,
    build_teacher_prompt,
    extract_hint,
)
from rllm.experimental.gsd.teacher_ref import FrozenTeacherRef
from rllm.experimental.rollout.rollout_engine import RolloutEngine
from rllm.workflows.workflow import Workflow

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type alias for the reward function: (task_dict, response_text) → float
RewardFn = Callable[[dict, str], float]


@dataclass
class GsdConfig:
    """Configuration for the GSD workflow."""

    # Core rollout sizes
    N: int = 4
    N_val: int = 2

    # Sampling parameters for the rollouts themselves (temperature, top_p,
    # max_tokens, etc.).  If empty, the engine's train_sampling_params are
    # used.
    rollout_sampling_params: dict[str, Any] = field(default_factory=dict)

    # Hint generation sampling params (typically lower temperature).
    hint_sampling_params: dict[str, Any] = field(
        default_factory=lambda: {"temperature": 1.0, "top_p": 0.95, "max_tokens": 512},
    )

    # Hint meta-RL: when True, the hint generation trajectory is added as
    # a ``gsd_hint`` trajectory optimized via REINFORCE / GRPO.  Requires
    # the custom grouping hook (``make_gsd_grouping_hook``) in the trainer.
    train_hint: bool = True

    # IS / reverse-KL advantage computation
    kl_coeff: float = 1.0
    kl_clip_min: float = -5.0
    kl_clip_max: float = 5.0
    success_reward_threshold: float = 0.5
    max_context_length: int = 32768

    # Experience buffer retrieval (for the math workflow's LLM hint generation)
    retrieval_k: int = 3
    # Threshold for collecting "hard solves" — tasks with student pass rate
    # below this that still have at least one correct solution.
    hard_solve_threshold: float = 0.5

    # When True, Case 2 (teacher invalid) produces NO training trajectories
    # on the student side.  Hint trajectories are still emitted when
    # ``train_hint=True`` (hints can get signal even when the hint itself
    # didn't help, via the negative reward).
    distill_only: bool = False


class GsdWorkflow(Workflow):
    """Generalized Self-Distillation workflow for single-turn tasks.

    Each :meth:`run` call processes one problem through the full GSD loop
    and returns an :class:`Episode` whose trajectories are dispatched to
    the correct loss by the per-role estimator map + custom transform.

    .. important::

       GSD manages its own rollout parallelism (``N`` student + ``N``
       teacher via ``sample_async(num_samples=N)``).  The Tinker
       ``training.group_size`` and ``validation.group_size`` should both be
       set to **1** in the config so the engine dispatches one task at a
       time to this workflow.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        *,
        reward_fn: RewardFn,
        gsd_config: GsdConfig | None = None,
        teacher_ref: FrozenTeacherRef | None = None,
        experience_store: Any | None = None,
        scoring_accumulator: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.cfg = gsd_config or GsdConfig()
        self.reward_fn = reward_fn
        # Shared across all workflow instances: pins the initial sampling
        # client as the frozen reference teacher on first access.
        self.teacher_ref = teacher_ref or FrozenTeacherRef()
        self.experience_store = experience_store  # Optional EmbeddingExperienceStore
        self.scoring_accumulator = scoring_accumulator  # Optional ScoringAccumulator

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs: Any) -> Episode:
        is_validation = self.rollout_engine.is_validation
        if is_validation:
            return await self._do_validation(task, uid)
        return await self._do_training(task, uid)

    # ------------------------------------------------------------------
    # Training: hint → dual rollouts → gate → distill / fallback
    # ------------------------------------------------------------------

    async def _do_training(self, task: dict, uid: str) -> Episode:
        question = task["question"]
        N = self.cfg.N

        # Capture the frozen teacher on first use.  No-op after first capture.
        teacher_client = self.teacher_ref.capture(self.rollout_engine)
        live_client: tinker.SamplingClient = self.rollout_engine.sampling_client

        # ---- Phase 1: Acquire a hint ------------------------------
        # ``_get_hint`` returns the hint text and optionally a ``Step``
        # representing the hint-generation trajectory (populated only when
        # the hint was sampled from the live model, so that it can receive
        # gradient signal via hint GRPO).  Subclasses that use a pool /
        # cached hint should return ``(hint_text, None)``.
        hint_text, hint_step = await self._get_hint(question, uid, live_client)

        # ---- Phase 2: Dual rollouts -------------------------------
        # Student uses the LIVE client, teacher uses the FROZEN client.
        # ``num_samples=N`` gives us N rollouts per call.
        student_messages = self._make_student_prompt(question)
        teacher_messages = self._make_teacher_prompt(question, hint_text)

        student_mi = self._build_model_input(student_messages)
        teacher_mi = self._build_model_input(teacher_messages)

        student_sp = self._make_sampling_params(self.cfg.rollout_sampling_params, prompt_len=student_mi.length)
        teacher_sp = self._make_sampling_params(self.cfg.rollout_sampling_params, prompt_len=teacher_mi.length)

        student_resp, teacher_resp = await asyncio.gather(
            live_client.sample_async(prompt=student_mi, num_samples=N, sampling_params=student_sp),
            teacher_client.sample_async(prompt=teacher_mi, num_samples=N, sampling_params=teacher_sp),
        )

        student_prompt_ids = student_mi.to_ints()
        teacher_prompt_ids = teacher_mi.to_ints()

        student_steps = [self._seq_to_step(student_prompt_ids, seq, task, student_messages) for seq in student_resp.sequences]
        teacher_steps = [self._seq_to_step(teacher_prompt_ids, seq, task, teacher_messages) for seq in teacher_resp.sequences]

        # ---- Phase 3: Gating --------------------------------------
        student_rewards = [s.reward for s in student_steps]
        teacher_rewards = [s.reward for s in teacher_steps]
        R_S_avg = sum(student_rewards) / max(len(student_rewards), 1)
        R_T_avg = sum(teacher_rewards) / max(len(teacher_rewards), 1)
        threshold = self.cfg.success_reward_threshold
        teacher_valid = R_T_avg > R_S_avg or (R_T_avg == R_S_avg and R_T_avg >= threshold)

        n_correct_student = sum(1 for r in student_rewards if r >= threshold)
        n_correct_teacher = sum(1 for r in teacher_rewards if r >= threshold)

        trajectories: list[Trajectory] = []
        metrics: dict[str, Any] = {
            "R_S_avg": R_S_avg,
            "R_T_avg": R_T_avg,
            "teacher_valid": float(teacher_valid),
            "hint_improvement": R_T_avg - R_S_avg,
            "n_correct_student": float(n_correct_student),
            "n_correct_teacher": float(n_correct_teacher),
        }

        # ---- Phase 4a: Hint trajectory ------------------------------
        # Only emitted when the hint was actually sampled from the live
        # model AND meta-RL on the hint generator is enabled.  Subclasses
        # that select hints from a pool (no sampling) return
        # ``hint_step is None`` from ``_get_hint``.
        if self.cfg.train_hint and hint_step is not None:
            hint_reward = R_T_avg - R_S_avg
            hint_step.reward = hint_reward
            trajectories.append(Trajectory(name=HINT_ROLE, steps=[hint_step], reward=hint_reward))

        # ---- Phase 4b: Case 1 (teacher valid) → CE + IS ------------
        if teacher_valid:
            logger.info(f"[{uid}] Case 1 (distill): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f} student={n_correct_student}/{N} teacher={n_correct_teacher}/{N}")
            await self._case1_distillation(
                uid=uid,
                teacher_client=teacher_client,
                student_prompt_ids=student_prompt_ids,
                teacher_prompt_ids=teacher_prompt_ids,
                student_messages=student_messages,
                student_steps=student_steps,
                teacher_steps=teacher_steps,
                trajectories=trajectories,
            )
        # ---- Phase 4c: Case 2 (teacher invalid) → GRPO fallback ----
        else:
            if self.cfg.distill_only:
                logger.info(f"[{uid}] Case 2 (skipped, distill_only=True): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f}")
            else:
                logger.info(f"[{uid}] Case 2 (GRPO fallback): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f} student={n_correct_student}/{N} teacher={n_correct_teacher}/{N}")
                for s in student_steps:
                    trajectories.append(Trajectory(name=GRPO_ROLE, steps=[s], reward=s.reward))

        # ---- Phase 5: Post-training hooks (experience buffer, etc.) ---
        await self._post_training_hook(
            task=task,
            uid=uid,
            question=question,
            hint_text=hint_text,
            student_steps=student_steps,
            teacher_steps=teacher_steps,
            R_S_avg=R_S_avg,
            R_T_avg=R_T_avg,
            teacher_valid=teacher_valid,
            metrics=metrics,
        )

        is_correct = max(student_rewards + teacher_rewards, default=0.0) >= threshold
        return self._build_episode(uid, task, trajectories, metrics, is_correct)

    # ------------------------------------------------------------------
    # Case 1: build CE and IS trajectories
    # ------------------------------------------------------------------

    async def _case1_distillation(
        self,
        *,
        uid: str,
        teacher_client: tinker.SamplingClient,
        student_prompt_ids: list[int],
        teacher_prompt_ids: list[int],
        student_messages: list[dict],
        student_steps: list[Step],
        teacher_steps: list[Step],
        trajectories: list[Trajectory],
    ) -> None:
        cfg = self.cfg

        # ---- (a) IS: teacher_lp on student responses (single call per student) ---
        # The frozen teacher evaluates its own logprobs on the student's
        # response tokens, conditioned on the teacher prompt (with hint).
        coros = [
            compute_teacher_logprobs_for_response(
                teacher_client,
                teacher_prompt_ids,
                s.response_ids,
                max_context_length=cfg.max_context_length,
            )
            for s in student_steps
        ]
        if not coros:
            teacher_lps_D = []
        elif self.scoring_accumulator is not None:
            teacher_lps_D = await asyncio.gather(*[self.scoring_accumulator.submit(c) for c in coros])
        else:
            teacher_lps_D = await asyncio.gather(*coros)

        is_count = 0
        for s, (teacher_lps, T) in zip(student_steps, teacher_lps_D, strict=True):
            if T == 0:
                continue
            student_lps = list(s.logprobs[:T])
            advantages = kl_advantages_from_logprobs(
                teacher_logprobs=teacher_lps,
                student_logprobs=student_lps,
                kl_coeff=cfg.kl_coeff,
                clip_min=cfg.kl_clip_min,
                clip_max=cfg.kl_clip_max,
            )
            # Truncate the response to T so advantage/response/logprobs stay aligned.
            is_step = Step(
                prompt_ids=list(student_prompt_ids),
                response_ids=list(s.response_ids[:T]),
                logprobs=student_lps,
                chat_completions=student_messages,
                model_response=s.model_response,
                reward=s.reward,
                advantage=advantages,
                done=True,
            )
            trajectories.append(Trajectory(name=IS_ROLE, steps=[is_step], reward=s.reward))
            is_count += 1

        # ---- (b) CE: teacher's correct responses prepended to the student prompt ---
        ce_count = 0
        for t in teacher_steps:
            if t.reward < cfg.success_reward_threshold:
                continue
            if not t.response_ids:
                continue
            ce_step = Step(
                prompt_ids=list(student_prompt_ids),
                response_ids=list(t.response_ids),
                logprobs=[],  # not used by cross_entropy
                chat_completions=student_messages,
                model_response=t.model_response,
                reward=t.reward,
                done=True,
            )
            trajectories.append(Trajectory(name=CE_ROLE, steps=[ce_step], reward=t.reward))
            ce_count += 1

        logger.info(f"[{uid}] distillation: {is_count} IS + {ce_count} CE trajectories")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    async def _do_validation(self, task: dict, uid: str) -> Episode:
        """Run validation rollouts and compute pass@N metrics for both roles."""
        question = task["question"]
        N = self.cfg.N_val

        teacher_client = self.teacher_ref.capture(self.rollout_engine)
        live_client = self.rollout_engine.sampling_client

        # Use the same ``_get_hint`` extension point as training so
        # subclasses get a single place to override.  We discard the
        # optional hint step — validation never produces training data.
        hint_text, _ = await self._get_hint(question, uid, live_client)

        student_messages = self._make_student_prompt(question)
        teacher_messages = self._make_teacher_prompt(question, hint_text)
        student_mi = self._build_model_input(student_messages)
        teacher_mi = self._build_model_input(teacher_messages)

        student_sp = self._make_sampling_params(self.cfg.rollout_sampling_params, prompt_len=student_mi.length)
        teacher_sp = self._make_sampling_params(self.cfg.rollout_sampling_params, prompt_len=teacher_mi.length)

        student_resp, teacher_resp = await asyncio.gather(
            live_client.sample_async(prompt=student_mi, num_samples=N, sampling_params=student_sp),
            teacher_client.sample_async(prompt=teacher_mi, num_samples=N, sampling_params=teacher_sp),
        )

        student_prompt_ids = student_mi.to_ints()
        teacher_prompt_ids = teacher_mi.to_ints()

        student_steps = [self._seq_to_step(student_prompt_ids, seq, task, student_messages) for seq in student_resp.sequences]
        teacher_steps = [self._seq_to_step(teacher_prompt_ids, seq, task, teacher_messages) for seq in teacher_resp.sequences]

        threshold = self.cfg.success_reward_threshold
        student_rewards = [s.reward for s in student_steps]
        teacher_rewards = [s.reward for s in teacher_steps]
        student_correct = [r >= threshold for r in student_rewards]
        teacher_correct = [r >= threshold for r in teacher_rewards]

        metrics: dict[str, Any] = {
            "val/student_pass@1": sum(student_correct) / N,
            "val/teacher_pass@1": sum(teacher_correct) / N,
            "val/hint_improvement": sum(teacher_correct) / N - sum(student_correct) / N,
            "val/student_any_correct": float(any(student_correct)),
            "val/teacher_any_correct": float(any(teacher_correct)),
            "val/student_avg_reward": sum(student_rewards) / N,
            "val/teacher_avg_reward": sum(teacher_rewards) / N,
        }

        logger.info(f"[{uid}] val: student_pass@1={sum(student_correct)}/{N} teacher_pass@1={sum(teacher_correct)}/{N} hint_improvement={sum(teacher_correct) / N - sum(student_correct) / N:+.2f}")

        # Return a single student trajectory for the trainer pipeline.
        traj = Trajectory(name=GRPO_ROLE, steps=[student_steps[0]], reward=student_rewards[0])
        return self._build_episode(uid, task, [traj], metrics, any(student_correct))

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _make_student_prompt(self, question: str) -> list[dict]:
        """Build the student prompt (no hint).  Override in subclasses."""
        return build_student_prompt(question)

    def _make_teacher_prompt(self, question: str, hint: str) -> list[dict]:
        """Build the teacher prompt (with hint).  Override in subclasses."""
        return build_teacher_prompt(question, hint)

    async def _build_hint_prompt_messages(self, question: str, uid: str) -> list[dict]:
        """Build the messages used to sample a hint.  Override in subclasses.

        The default implementation calls :func:`build_hint_prompt`, optionally
        augmented with similar past experiences retrieved from the workflow's
        experience store.
        """
        experiences = None
        if self.experience_store is not None:
            experiences = await self.experience_store.query(question, top_k=self.cfg.retrieval_k)
            if experiences:
                logger.info(f"[{uid}] retrieved {len(experiences)} experiences (store size={self.experience_store.size})")
        return build_hint_prompt(question, experiences=experiences)

    async def _get_hint(
        self,
        question: str,
        uid: str,
        live_client: tinker.SamplingClient,
    ) -> tuple[str, Step | None]:
        """Return ``(hint_text, hint_step)`` for the current task.

        The default implementation samples a hint from the live model via
        ``sample_async`` and returns the fully populated ``Step`` so it can
        later become a ``gsd_hint`` trajectory for GRPO.

        Subclasses that use a different hint source (e.g. a shared
        :class:`HintPool`) should override this to return
        ``(hint_text, None)`` — a ``None`` step means no gradient flows
        to the hint generator for this task.
        """
        messages = await self._build_hint_prompt_messages(question, uid)
        hint_mi = self._build_model_input(messages)
        sp = self._make_sampling_params(self.cfg.hint_sampling_params, prompt_len=hint_mi.length)
        resp = await live_client.sample_async(
            prompt=hint_mi,
            num_samples=1,
            sampling_params=sp,
        )
        seq = resp.sequences[0]
        raw_text = self._decode(list(seq.tokens))
        hint_text = extract_hint(raw_text)
        logger.info(f"[{uid}] hint generated ({len(hint_text)} chars): {hint_text[:200]}")

        hint_step = Step(
            prompt_ids=list(hint_mi.to_ints()),
            response_ids=list(seq.tokens),
            logprobs=list(seq.logprobs),
            chat_completions=messages,
            model_response=raw_text,
            reward=0.0,  # overwritten later with R_T - R_S
            done=True,
        )
        return hint_text, hint_step

    async def _post_training_hook(
        self,
        *,
        task: dict,
        uid: str,
        question: str,
        hint_text: str,
        student_steps: list[Step],
        teacher_steps: list[Step],
        R_S_avg: float,
        R_T_avg: float,
        teacher_valid: bool,
        metrics: dict[str, Any],
    ) -> None:
        """Post-training hook — default implementation updates the experience buffer.

        Subclasses (e.g. ``GsdCountdownWorkflow``) override this to update
        a shared ``HintPool`` instead.
        """
        if teacher_valid and self.experience_store is not None:
            n_correct_teacher = sum(1 for s in teacher_steps if s.reward >= self.cfg.success_reward_threshold)
            await self.experience_store.add(
                text=question,
                metadata={
                    "hint": hint_text,
                    "summary": (f"Teacher outperformed student ({R_T_avg:.2f} vs {R_S_avg:.2f}). {n_correct_teacher}/{len(teacher_steps)} teacher rollouts correct."),
                    "improvement": R_T_avg - R_S_avg,
                },
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model_input(self, messages: list[dict]) -> tinker.ModelInput:
        """Build a :class:`tinker.ModelInput` from OpenAI-style messages.

        Uses the renderer's ``build_prompt`` path so there is no
        ``parse → string → encode`` round-trip and ImageChunks are preserved.
        """
        return self.rollout_engine.chat_parser.build_prompt(messages)

    def _decode(self, token_ids: list[int]) -> str:
        """Decode a token sequence to text for reward scoring and logging."""
        return self.rollout_engine.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _make_sampling_params(self, overrides: dict[str, Any], prompt_len: int) -> tinker.SamplingParams:
        """Build a :class:`tinker.SamplingParams` from the engine defaults.

        Starts from the engine's train/val sampling params, applies the
        caller's overrides, and clamps ``max_tokens`` so that
        ``prompt_len + max_tokens`` stays within ``max_model_length``.
        """
        engine = self.rollout_engine
        base = engine.val_sampling_params.copy() if engine.is_validation else engine.train_sampling_params.copy()
        base.update(overrides or {})

        # Resolve max_tokens with the same precedence the engine uses in
        # ``_prepare_max_tokens``.
        max_tokens = base.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = getattr(engine, "max_response_length", 4096)
        max_model_length = getattr(engine, "max_model_length", None)
        if max_model_length:
            remaining = max_model_length - prompt_len
            if remaining <= 0:
                max_tokens = 1  # degenerate case — let the server truncate
            elif remaining < max_tokens:
                max_tokens = remaining

        # Stop sequences from the chat parser.
        stop = list(getattr(engine, "stop_sequences", []) or [])

        return tinker.SamplingParams(
            max_tokens=int(max_tokens),
            stop=stop,
            **base,
        )

    def _seq_to_step(
        self,
        prompt_ids: list[int],
        seq: Any,  # tinker.SampledSequence
        task: dict,
        messages: list[dict],
    ) -> Step:
        """Convert a ``SampledSequence`` into a :class:`Step` with its reward.

        We build the Step by hand so we can skip the
        ``ModelOutput → Step.from_model_output`` round-trip.  The fields we
        populate are the only ones the downstream transform / datum builders
        actually read: ``prompt_ids``, ``response_ids``, ``logprobs``,
        ``reward``, ``done``, ``chat_completions`` (for logging), and
        ``model_response`` (decoded text).
        """
        response_ids = list(seq.tokens)
        logprobs = list(seq.logprobs)
        response_text = self._decode(response_ids)
        reward = float(self.reward_fn(task, response_text))
        return Step(
            prompt_ids=list(prompt_ids),
            response_ids=response_ids,
            logprobs=logprobs,
            chat_completions=messages,
            model_response=response_text,
            reward=reward,
            done=True,
        )

    def _build_episode(
        self,
        uid: str,
        task: dict,
        trajectories: list[Trajectory],
        metrics: dict,
        is_correct: bool,
    ) -> Episode:
        return Episode(
            id=uid,
            task=task,
            is_correct=is_correct,
            trajectories=trajectories,
            metrics=metrics,
        )
