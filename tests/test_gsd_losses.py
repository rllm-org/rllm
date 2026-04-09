"""Tests for ``rllm.experimental.gsd`` loss utilities and the legacy archive.

The tests in this file are split into two sections:

* ``TestLegacy*`` — tests for the archived top-K CE + combined custom-loss
  variant living under ``rllm.experimental.gsd.legacy``.
* Everything else — tests for the active GSD pipeline (SFT-style CE + IS
  datum builders, advantage helpers, grouping hook, etc.).
"""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# ===========================================================================
# LEGACY TESTS — exercise the archived top-K pipeline
# ===========================================================================
from rllm.experimental.gsd.legacy.losses_topk import (
    DEFAULT_GSD_ADV_ESTIMATOR_MAP,
    compute_sampled_rkl_advantages,
    compute_student_logprobs_for_teacher_topk,
    compute_topk_rkl_at_position,
)
from rllm.experimental.gsd.legacy.losses_topk import (
    build_gsd_estimator_map as build_legacy_gsd_estimator_map,
)


class TestLegacyRenormalization:
    """Top-K logprobs are correctly renormalized to probabilities."""

    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def test_weights_sum_to_one(self):
        from rllm.experimental.gsd.legacy.losses_topk import build_topk_fkl_datum

        teacher_topk = {
            "topk_ids": [[10, 20, 30]],
            "topk_logprobs": [[-1.0, -2.0, -3.0]],
        }
        datum = build_topk_fkl_datum([1, 2], [3], teacher_topk)
        weights = datum.loss_fn_inputs["weights"].to_torch()
        resp_weights = weights[-1]
        np.testing.assert_allclose(resp_weights.sum().item(), 1.0, atol=1e-6)

    def test_inf_padding_ignored(self):
        from rllm.experimental.gsd.legacy.losses_topk import build_topk_fkl_datum

        teacher_topk = {
            "topk_ids": [[10, 0]],
            "topk_logprobs": [[-0.5, float("-inf")]],
        }
        datum = build_topk_fkl_datum([1], [2], teacher_topk)
        weights = datum.loss_fn_inputs["weights"].to_torch()
        np.testing.assert_allclose(weights[-1, 0].item(), 1.0, atol=1e-6)
        assert weights[-1, 1].item() == 0.0


class TestLegacySampledRklAdvantages:
    def test_basic(self):
        teacher = [-0.5, -1.0, -2.0]
        student = [-1.0, -0.5, -2.0]
        result = compute_sampled_rkl_advantages(teacher, student, kl_coeff=1.0)
        expected = [0.5, -0.5, 0.0]
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_kl_coeff(self):
        teacher = [-0.5, -1.0]
        student = [-1.0, -0.5]
        result = compute_sampled_rkl_advantages(teacher, student, kl_coeff=2.0)
        expected = [1.0, -1.0]
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_clipping(self):
        teacher = [-0.1]
        student = [-10.0]
        result = compute_sampled_rkl_advantages(teacher, student, kl_coeff=1.0, clip_min=-5.0, clip_max=5.0)
        assert result[0] == pytest.approx(5.0, abs=1e-8)


class TestLegacyTopkRklAtPosition:
    def test_identical_distributions(self):
        lps = [math.log(0.7), math.log(0.2), math.log(0.1)]
        assert compute_topk_rkl_at_position(lps, lps) == pytest.approx(0.0, abs=1e-8)

    def test_positive_for_different_distributions(self):
        teacher = [math.log(0.8), math.log(0.15), math.log(0.05)]
        student = [math.log(0.4), math.log(0.3), math.log(0.3)]
        assert compute_topk_rkl_at_position(teacher, student) > 0.0

    def test_known_value(self):
        t = [math.log(0.5), math.log(0.5)]
        s = [math.log(0.8), math.log(0.2)]
        expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
        assert compute_topk_rkl_at_position(t, s) == pytest.approx(expected, abs=1e-6)


class TestLegacyStudentLogprobsForTeacherTopk:
    def _make_mock_client(self, topk_data, scalar_data):
        client = AsyncMock()

        async def fake_sample(**kwargs):
            result = MagicMock()
            result.topk_prompt_logprobs = topk_data
            result.prompt_logprobs = scalar_data
            return result

        client.sample_async = fake_sample
        return client

    def test_full_coverage(self):
        teacher_topk = {
            "topk_ids": [[10, 20]],
            "topk_logprobs": [[-0.5, -1.0]],
        }
        student_topk_raw = [
            None,
            [(1, -0.1)],
            [(2, -0.2)],
            [(10, -0.3), (20, -0.7), (30, -1.5)],
        ]
        scalar_raw = [None, -0.1, -0.2, -0.4]
        client = self._make_mock_client(student_topk_raw, scalar_raw)

        result = asyncio.get_event_loop().run_until_complete(
            compute_student_logprobs_for_teacher_topk(
                sampling_client=client,
                student_prompt_ids=[1, 2, 3],
                response_ids=[4],
                teacher_topk=teacher_topk,
                student_k=3,
            )
        )
        assert len(result["logprobs"]) == 1
        np.testing.assert_allclose(result["logprobs"][0], [-0.3, -0.7], atol=1e-8)

    def test_partial_coverage_uses_floor(self):
        teacher_topk = {
            "topk_ids": [[10, 99]],
            "topk_logprobs": [[-0.5, -1.0]],
        }
        student_topk_raw = [None, [(10, -0.3), (20, -0.7)]]
        scalar_raw = [None, -0.4]
        client = self._make_mock_client(student_topk_raw, scalar_raw)

        floor = -18.42
        result = asyncio.get_event_loop().run_until_complete(
            compute_student_logprobs_for_teacher_topk(
                sampling_client=client,
                student_prompt_ids=[1],
                response_ids=[4],
                teacher_topk=teacher_topk,
                student_k=2,
                floor_logprob=floor,
            )
        )
        assert result["logprobs"][0][0] == pytest.approx(-0.3)
        assert result["logprobs"][0][1] == pytest.approx(floor)


class TestLegacyEstimatorMap:
    def test_default_map_keys(self):
        assert set(DEFAULT_GSD_ADV_ESTIMATOR_MAP.keys()) == {
            "gsd_student",
            "gsd_distill_onpolicy",
            "gsd_distill_supervised",
            "gsd_hint",
        }

    def test_build_has_combined_distill(self):
        m = build_legacy_gsd_estimator_map(train_hint=False)
        assert "gsd_distill" in m
        estimator, loss_fn = m["gsd_distill"]
        assert estimator == "precomputed"
        assert callable(loss_fn)

    def test_build_student_role(self):
        from rllm.experimental.common.config import rLLMAdvantageEstimator

        m = build_legacy_gsd_estimator_map(train_hint=False)
        estimator, loss = m["gsd_student"]
        assert estimator == rLLMAdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        assert loss == "ppo"

    def test_build_without_hint(self):
        assert "gsd_hint" not in build_legacy_gsd_estimator_map(train_hint=False)

    def test_build_with_hint(self):
        assert "gsd_hint" in build_legacy_gsd_estimator_map(train_hint=True)

    def test_callable_flows_through_algorithm_config(self):
        from rllm.experimental.common.config import AlgorithmConfig

        m = build_legacy_gsd_estimator_map(train_hint=False)
        config = AlgorithmConfig(estimator_map=m)
        assert callable(config.loss_fn_map["gsd_distill"])


class TestLegacyBuildTopkFklDatum:
    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def test_datum_shape_and_mask(self):
        from rllm.experimental.gsd.legacy.losses_topk import build_topk_fkl_datum

        prompt_ids = [1, 2, 3]
        response_ids = [4, 5]
        K = 3
        teacher_topk = {
            "topk_ids": [[10, 20, 30], [40, 50, 60]],
            "topk_logprobs": [[-0.5, -1.0, -2.0], [-0.3, -0.8, -1.5]],
        }
        datum = build_topk_fkl_datum(prompt_ids, response_ids, teacher_topk)

        target_t = datum.loss_fn_inputs["target_tokens"].to_torch()
        weights_t = datum.loss_fn_inputs["weights"].to_torch()
        mask_t = datum.loss_fn_inputs["mask"].to_torch()

        N = len(prompt_ids) + len(response_ids) - 1
        assert target_t.shape == (N, K)
        assert weights_t.shape == (N, K)
        assert mask_t.shape == (N,)

        prompt_len = N - len(response_ids)
        for t in range(prompt_len):
            assert target_t[t].sum().item() == 0
            assert weights_t[t].sum().item() == 0.0
            assert mask_t[t].item() == 0.0

        resp_start = prompt_len
        assert target_t[resp_start].tolist() == [10, 20, 30]
        assert mask_t[resp_start].item() == 1.0
        np.testing.assert_allclose(weights_t[resp_start].sum().item(), 1.0, atol=1e-6)

    def test_weight_clamp(self):
        from rllm.experimental.gsd.legacy.losses_topk import build_topk_fkl_datum

        teacher_topk = {
            "topk_ids": [[10, 20]],
            "topk_logprobs": [[math.log(0.99), math.log(0.01)]],
        }
        datum = build_topk_fkl_datum([1], [2], teacher_topk, loss_clamp=0.5)
        weights_t = datum.loss_fn_inputs["weights"].to_torch()
        assert weights_t.max().item() <= 0.5 + 1e-8


class TestLegacyBuildCombinedGsdDatum:
    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def test_shape_with_teacher_topk(self):
        from rllm.experimental.gsd.legacy.losses_topk import build_combined_gsd_datum

        prompt_ids = [1, 2]
        response_ids = [3, 4]
        K = 3
        teacher_topk = {
            "topk_ids": [[10, 20, 30], [40, 50, 60]],
            "topk_logprobs": [[-0.5, -1.0, -2.0], [-0.3, -0.8, -1.5]],
        }
        datum = build_combined_gsd_datum(
            prompt_ids,
            response_ids,
            teacher_topk=teacher_topk,
            is_advantages=[0.1, 0.2],
            is_old_logprobs=[-1.0, -2.0],
            K=K,
        )

        target_t = datum.loss_fn_inputs["target_tokens"].to_torch()
        meta = datum._gsd_metadata
        weights_t = meta["teacher_weights"]
        adv_t = meta["is_advantages"]
        mask_t = meta["mask"]

        N = len(prompt_ids) + len(response_ids) - 1
        assert target_t.shape == (N, K + 1)
        assert weights_t.shape == (N, K + 1)
        assert adv_t.shape == (N,)
        assert mask_t.shape == (N,)

        assert target_t[-1, K].item() != 0

        prompt_len = N - len(response_ids)
        resp_weights = weights_t[prompt_len, :K]
        np.testing.assert_allclose(resp_weights.sum().item(), 1.0, atol=1e-5)
        assert weights_t[:, K].sum().item() == 0.0

    def test_onpolicy_only_no_teacher(self):
        from rllm.experimental.gsd.legacy.losses_topk import build_combined_gsd_datum

        datum = build_combined_gsd_datum(
            prompt_ids=[1],
            response_ids=[2, 3],
            teacher_topk=None,
            is_advantages=[0.5, -0.3],
            is_old_logprobs=[-1.0, -2.0],
            K=5,
        )
        meta = datum._gsd_metadata
        assert meta["teacher_weights"].sum().item() == 0.0
        assert meta["is_advantages"].sum().item() != 0.0


class TestLegacyGsdCombinedLoss:
    def test_loss_runs_and_returns_metrics(self):
        import torch

        from rllm.experimental.gsd.legacy.losses_topk import make_gsd_combined_loss

        loss_fn = make_gsd_combined_loss(ce_weight=0.5, is_weight=0.5)

        class MockDatum:
            def __init__(self, N, K):
                self.loss_fn_inputs = {"target_tokens": None}
                self._gsd_metadata = {
                    "teacher_weights": torch.zeros(N, K + 1),
                    "is_advantages": torch.tensor([0.5] + [0.0] * (N - 1)),
                    "is_old_logprobs": torch.tensor([-1.0] + [0.0] * (N - 1)),
                    "mask": torch.tensor([1.0] + [0.0] * (N - 1)),
                }

        N, K = 4, 3
        datum = MockDatum(N, K)
        logprobs = torch.randn(N, K + 1, requires_grad=True)

        loss, metrics = loss_fn([datum], [logprobs])

        assert loss.shape == ()
        assert loss.requires_grad
        assert "gsd/combined_loss" in metrics
        assert "gsd/ce_loss" in metrics
        assert "gsd/is_loss" in metrics


# ===========================================================================
# NEW PIPELINE TESTS — SFT-style CE + IS + grouping hook + teacher ref
# ===========================================================================

from rllm.experimental.gsd import (  # noqa: E402 (imports after legacy block for clarity)
    CE_ROLE,
    GRPO_ROLE,
    GSD_ROLES,
    HINT_ROLE,
    IS_ROLE,
    FrozenTeacherRef,
    build_gsd_estimator_map,
    build_is_datum,
    build_sft_style_ce_datum,
    kl_advantages_from_logprobs,
    make_gsd_grouping_hook,
)


class TestNewEstimatorMap:
    def test_roles_present(self):
        m = build_gsd_estimator_map(train_hint=True)
        assert set(m.keys()) == {CE_ROLE, IS_ROLE, GRPO_ROLE, HINT_ROLE}

    def test_train_hint_false_omits_hint(self):
        m = build_gsd_estimator_map(train_hint=False)
        assert HINT_ROLE not in m
        assert CE_ROLE in m and IS_ROLE in m and GRPO_ROLE in m

    def test_ce_role_uses_cross_entropy(self):
        m = build_gsd_estimator_map()
        estimator, loss_fn = m[CE_ROLE]
        assert estimator == "precomputed"
        assert loss_fn == "cross_entropy"

    def test_is_role_uses_importance_sampling(self):
        m = build_gsd_estimator_map()
        estimator, loss_fn = m[IS_ROLE]
        assert estimator == "precomputed"
        assert loss_fn == "importance_sampling"

    def test_grpo_role_uses_ppo(self):
        from rllm.experimental.common.config import rLLMAdvantageEstimator

        m = build_gsd_estimator_map()
        estimator, loss_fn = m[GRPO_ROLE]
        assert estimator == rLLMAdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        assert loss_fn == "ppo"

    def test_flows_through_algorithm_config(self):
        """The new estimator map has only string losses, so no callable plumbing issues."""
        from rllm.experimental.common.config import AlgorithmConfig

        m = build_gsd_estimator_map(train_hint=True)
        config = AlgorithmConfig(estimator_map=m)
        # All loss entries are strings (no custom callables in the new pipeline)
        for role in GSD_ROLES:
            loss = config.loss_fn_map[role]
            assert isinstance(loss, str), f"role {role} loss is {type(loss)}"


class TestKlAdvantagesFromLogprobs:
    def test_basic_difference(self):
        teacher = [-0.5, -1.0, -2.0]
        student = [-1.0, -0.5, -2.0]
        result = kl_advantages_from_logprobs(teacher, student, kl_coeff=1.0)
        np.testing.assert_allclose(result, [0.5, -0.5, 0.0], atol=1e-8)

    def test_kl_coeff_scales(self):
        teacher = [-0.5, -1.0]
        student = [-1.0, -0.5]
        result = kl_advantages_from_logprobs(teacher, student, kl_coeff=2.0)
        np.testing.assert_allclose(result, [1.0, -1.0], atol=1e-8)

    def test_clipping(self):
        teacher = [-0.1]
        student = [-10.0]
        result = kl_advantages_from_logprobs(teacher, student, kl_coeff=1.0, clip_min=-5.0, clip_max=5.0)
        assert result[0] == pytest.approx(5.0, abs=1e-8)

    def test_length_mismatch_uses_min(self):
        # Short teacher sequence — only first 2 positions are computed.
        teacher = [-0.1, -0.2]
        student = [-0.3, -0.4, -0.5]
        result = kl_advantages_from_logprobs(teacher, student)
        assert len(result) == 2


class TestBuildSftStyleCeDatum:
    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def test_shapes_and_mask(self):
        prompt_ids = [1, 2, 3, 4]
        response_ids = [5, 6, 7]
        datum = build_sft_style_ce_datum(prompt_ids, response_ids)

        # After right-shift: N = len(prompt) + len(response) - 1 = 6
        N = len(prompt_ids) + len(response_ids) - 1
        assert datum.model_input.length == N

        target_t = datum.loss_fn_inputs["target_tokens"].to_torch()
        weights_t = datum.loss_fn_inputs["weights"].to_torch()

        assert target_t.shape == (N,)
        assert weights_t.shape == (N,)

    def test_weights_mask_response_only(self):
        prompt_ids = [1, 2, 3, 4]
        response_ids = [5, 6, 7]
        datum = build_sft_style_ce_datum(prompt_ids, response_ids)
        weights = datum.loss_fn_inputs["weights"].to_torch().tolist()
        # Prompt positions (first 3 of 6) are 0; response positions (last 3) are 1.
        assert weights == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    def test_targets_are_left_shifted(self):
        """``target_tokens`` is the full sequence minus the first token."""
        prompt_ids = [1, 2, 3]
        response_ids = [4, 5]
        datum = build_sft_style_ce_datum(prompt_ids, response_ids)
        targets = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
        # Full sequence: [1,2,3,4,5]; left-shifted: [2,3,4,5]
        assert targets == [2, 3, 4, 5]


class TestBuildIsDatum:
    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def test_shapes(self):
        datum = build_is_datum(
            prompt_ids=[1, 2, 3],
            response_ids=[4, 5],
            logprobs=[-0.1, -0.2],
            advantages=[0.5, 0.3],
        )
        N = 3 + 2 - 1
        assert datum.model_input.length == N
        for key in ("target_tokens", "logprobs", "advantages", "mask"):
            t = datum.loss_fn_inputs[key].to_torch()
            assert t.shape == (N,), f"{key} has shape {t.shape}"

    def test_masked_fields_align_on_response(self):
        prompt_ids = [1, 2, 3, 4]
        response_ids = [5, 6, 7]
        logprobs = [-0.1, -0.2, -0.3]
        advantages = [0.5, 0.25, -0.1]

        datum = build_is_datum(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=logprobs,
            advantages=advantages,
        )
        mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
        lps = datum.loss_fn_inputs["logprobs"].to_torch().tolist()
        advs = datum.loss_fn_inputs["advantages"].to_torch().tolist()

        assert mask == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        # Prompt positions are 0; response positions carry the true values
        np.testing.assert_allclose(lps[:3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(lps[3:], logprobs, atol=1e-6)
        np.testing.assert_allclose(advs[:3], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(advs[3:], advantages, atol=1e-6)

    def test_scalar_advantage_broadcasts(self):
        datum = build_is_datum(
            prompt_ids=[1, 2],
            response_ids=[3, 4, 5],
            logprobs=[-0.1, -0.2, -0.3],
            advantages=0.7,
        )
        advs = datum.loss_fn_inputs["advantages"].to_torch().tolist()
        # N = 2 + 3 - 1 = 4, prompt_len = 1 → [0, 0.7, 0.7, 0.7]
        np.testing.assert_allclose(advs, [0.0, 0.7, 0.7, 0.7], atol=1e-6)


class TestFrozenTeacherRef:
    def test_capture_installs_reference(self):
        ref = FrozenTeacherRef()
        assert not ref.is_captured

        class FakeEngine:
            sampling_client = object()

        engine = FakeEngine()
        captured = ref.capture(engine)
        assert ref.is_captured
        assert captured is engine.sampling_client
        assert ref.client is engine.sampling_client

    def test_capture_is_idempotent(self):
        ref = FrozenTeacherRef()

        class FakeEngine:
            sampling_client = object()

        engine = FakeEngine()
        first = ref.capture(engine)
        # Swap out the engine's client (simulating a post-optim refresh)
        new_client = object()
        engine.sampling_client = new_client
        second = ref.capture(engine)
        # Second capture is a no-op: we still hold the original client.
        assert first is second
        assert second is not new_client

    def test_client_before_capture_raises(self):
        ref = FrozenTeacherRef()
        with pytest.raises(RuntimeError):
            _ = ref.client

    def test_capture_without_sampling_client_raises(self):
        ref = FrozenTeacherRef()

        class FakeEngine:
            sampling_client = None

        with pytest.raises(RuntimeError):
            ref.capture(FakeEngine())


class TestMakeGsdGroupingHook:
    """The grouping hook merges hint trajectories across tasks."""

    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def _make_episodes(self, n_tasks: int = 3, include_non_hint: bool = True):
        """Build ``n_tasks`` episodes, each with one hint trajectory and
        optionally one non-hint (``gsd_is``) trajectory."""
        from rllm.agents.agent import Episode, Step, Trajectory
        from rllm.workflows.workflow import TerminationReason

        episodes = []
        for i in range(n_tasks):
            hint_step = Step(
                prompt_ids=[1, 2],
                response_ids=[10 + i, 20 + i],
                logprobs=[-0.1, -0.2],
                reward=float(i) / 10.0,
                done=True,
            )
            trajs = [
                Trajectory(
                    name=HINT_ROLE,
                    steps=[hint_step],
                    reward=float(i) / 10.0,
                )
            ]
            if include_non_hint:
                is_step = Step(
                    prompt_ids=[3, 4],
                    response_ids=[50 + i],
                    logprobs=[-0.3],
                    advantage=[0.1],
                    reward=1.0,
                    done=True,
                )
                trajs.append(Trajectory(name=IS_ROLE, steps=[is_step], reward=1.0))
            episodes.append(
                Episode(
                    id=f"task_{i}:0",
                    task={"question": f"q{i}"},
                    trajectories=trajs,
                    termination_reason=TerminationReason.UNKNOWN,
                    is_correct=False,
                )
            )
        return episodes

    def _make_transform_config(self):
        from rllm.experimental.common.config import TransformConfig

        return TransformConfig()

    def test_hint_trajectories_merged_into_single_group(self):
        hook = make_gsd_grouping_hook(hint_role=HINT_ROLE)
        episodes = self._make_episodes(n_tasks=3, include_non_hint=False)
        groups = hook(episodes, self._make_transform_config())

        hint_groups = [g for g in groups if g.group_role == HINT_ROLE]
        assert len(hint_groups) == 1
        assert len(hint_groups[0].trajectories) == 3
        assert hint_groups[0].group_id == "cross_task:gsd_hint"

    def test_non_hint_groups_unchanged(self):
        hook = make_gsd_grouping_hook(hint_role=HINT_ROLE)
        episodes = self._make_episodes(n_tasks=3, include_non_hint=True)
        groups = hook(episodes, self._make_transform_config())

        # One merged hint group + one gsd_is group per task = 4 groups total
        assert len(groups) == 4
        is_groups = [g for g in groups if g.group_role == IS_ROLE]
        assert len(is_groups) == 3  # one per task_id
        for g in is_groups:
            assert len(g.trajectories) == 1

    def test_empty_hint_no_merge(self):
        """With no hint trajectories at all, the hook returns the default groups as-is."""
        hook = make_gsd_grouping_hook(hint_role=HINT_ROLE)
        episodes = self._make_episodes(n_tasks=2, include_non_hint=True)
        # Strip the hint trajectories
        for ep in episodes:
            ep.trajectories = [t for t in ep.trajectories if t.name != HINT_ROLE]

        groups = hook(episodes, self._make_transform_config())
        assert all(g.group_role != HINT_ROLE for g in groups)
        # 2 tasks × 1 non-hint role = 2 groups
        assert len(groups) == 2
