"""Tests for the rllm.experimental.gsd.losses module."""

from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from rllm.experimental.gsd.losses import (
    DEFAULT_GSD_ADV_ESTIMATOR_MAP,
    build_gsd_estimator_map,
    compute_sampled_rkl_advantages,
    compute_student_logprobs_for_teacher_topk,
    compute_topk_rkl_at_position,
)

# ---------------------------------------------------------------------------
# Renormalization (tested indirectly through build_topk_fkl_datum weights)
# ---------------------------------------------------------------------------


class TestRenormalization:
    """Verify that teacher logprobs are correctly renormalized to probabilities."""

    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def test_weights_sum_to_one(self):
        """Renormalized Top-K weights for each response position sum to 1."""
        from rllm.experimental.gsd.losses import build_topk_fkl_datum

        teacher_topk = {
            "topk_ids": [[10, 20, 30]],
            "topk_logprobs": [[-1.0, -2.0, -3.0]],  # unnormalized
        }
        datum = build_topk_fkl_datum([1, 2], [3], teacher_topk)
        weights = datum.loss_fn_inputs["weights"].to_torch()
        # Only the response position (last row) has non-zero weights
        resp_weights = weights[-1]
        np.testing.assert_allclose(resp_weights.sum().item(), 1.0, atol=1e-6)

    def test_inf_padding_ignored(self):
        """-inf padded entries get weight 0 after renormalization."""
        from rllm.experimental.gsd.losses import build_topk_fkl_datum

        teacher_topk = {
            "topk_ids": [[10, 0]],
            "topk_logprobs": [[-0.5, float("-inf")]],
        }
        datum = build_topk_fkl_datum([1], [2], teacher_topk)
        weights = datum.loss_fn_inputs["weights"].to_torch()
        # Single valid token gets weight 1.0, padded gets 0.0
        np.testing.assert_allclose(weights[-1, 0].item(), 1.0, atol=1e-6)
        assert weights[-1, 1].item() == 0.0


# ---------------------------------------------------------------------------
# compute_sampled_rkl_advantages (uses compute_distill_reverse_kl)
# ---------------------------------------------------------------------------


class TestSampledRklAdvantages:
    def test_basic(self):
        """advantage = kl_coeff * (teacher_lp - student_lp)."""
        teacher = [-0.5, -1.0, -2.0]
        student = [-1.0, -0.5, -2.0]
        result = compute_sampled_rkl_advantages(teacher, student, kl_coeff=1.0)
        expected = [0.5, -0.5, 0.0]
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_kl_coeff(self):
        """Coefficient scales advantages."""
        teacher = [-0.5, -1.0]
        student = [-1.0, -0.5]
        result = compute_sampled_rkl_advantages(teacher, student, kl_coeff=2.0)
        expected = [1.0, -1.0]
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_clipping(self):
        """Advantages are clipped to [clip_min, clip_max]."""
        teacher = [-0.1]
        student = [-10.0]  # huge gap
        result = compute_sampled_rkl_advantages(
            teacher,
            student,
            kl_coeff=1.0,
            clip_min=-5.0,
            clip_max=5.0,
        )
        assert result[0] == pytest.approx(5.0, abs=1e-8)


# ---------------------------------------------------------------------------
# compute_topk_rkl_at_position
# ---------------------------------------------------------------------------


class TestTopkRklAtPosition:
    def test_identical_distributions(self):
        """RKL between identical distributions is 0."""
        lps = [math.log(0.7), math.log(0.2), math.log(0.1)]
        assert compute_topk_rkl_at_position(lps, lps) == pytest.approx(0.0, abs=1e-8)

    def test_positive_for_different_distributions(self):
        """RKL is positive when distributions differ."""
        teacher = [math.log(0.8), math.log(0.15), math.log(0.05)]
        student = [math.log(0.4), math.log(0.3), math.log(0.3)]
        rkl = compute_topk_rkl_at_position(teacher, student)
        assert rkl > 0.0

    def test_known_value(self):
        """Check against hand-computed RKL for a simple case.

        Teacher (renormalized): [0.5, 0.5], Student (renormalized): [0.8, 0.2]
        RKL(student || teacher) = 0.8 * log(0.8/0.5) + 0.2 * log(0.2/0.5)
        """
        t = [math.log(0.5), math.log(0.5)]
        s = [math.log(0.8), math.log(0.2)]
        expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
        assert compute_topk_rkl_at_position(t, s) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_student_logprobs_for_teacher_topk
# ---------------------------------------------------------------------------


class TestStudentLogprobsForTeacherTopk:
    """Tests for the student Top-K lookup utility."""

    def _make_mock_client(self, topk_data, scalar_data):
        """Build a mock sampling_client that returns canned scoring data."""
        client = AsyncMock()

        async def fake_sample(**kwargs):
            result = MagicMock()
            result.topk_prompt_logprobs = topk_data
            result.prompt_logprobs = scalar_data
            return result

        client.sample_async = fake_sample
        return client

    def test_full_coverage(self):
        """When student's Top-K covers all teacher tokens."""
        teacher_topk = {
            "topk_ids": [[10, 20]],
            "topk_logprobs": [[-0.5, -1.0]],
        }

        # Student scoring: prompt=[1,2,3], response=[4]
        # Position 3 (prompt_len=3) is the response position
        student_topk_raw = [
            None,  # pos 0 (BOS)
            [(1, -0.1)],  # pos 1
            [(2, -0.2)],  # pos 2
            [(10, -0.3), (20, -0.7), (30, -1.5)],  # pos 3 (response)
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
        """When teacher token is not in student's Top-K, use floor logprob."""
        teacher_topk = {
            "topk_ids": [[10, 99]],  # 99 not in student's Top-K
            "topk_logprobs": [[-0.5, -1.0]],
        }
        student_topk_raw = [
            None,
            [(10, -0.3), (20, -0.7)],
        ]
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


# ---------------------------------------------------------------------------
# Estimator map
# ---------------------------------------------------------------------------


class TestEstimatorMap:
    def test_default_map_keys(self):
        assert set(DEFAULT_GSD_ADV_ESTIMATOR_MAP.keys()) == {
            "gsd_student",
            "gsd_distill_onpolicy",
            "gsd_distill_supervised",
            "gsd_hint",
        }

    def test_on_policy_uses_importance_sampling(self):
        """On-policy distillation routes to importance_sampling (reverse KL)."""
        _, loss = DEFAULT_GSD_ADV_ESTIMATOR_MAP["gsd_distill_onpolicy"]
        assert loss == "importance_sampling"

    def test_supervised_uses_cross_entropy(self):
        """Supervised distillation routes to cross_entropy (forward KL)."""
        _, loss = DEFAULT_GSD_ADV_ESTIMATOR_MAP["gsd_distill_supervised"]
        assert loss == "cross_entropy"

    def test_student_uses_grpo(self):
        from rllm.experimental.common.config import rLLMAdvantageEstimator

        estimator, loss = DEFAULT_GSD_ADV_ESTIMATOR_MAP["gsd_student"]
        assert estimator == rLLMAdvantageEstimator.GRPO
        assert loss == "importance_sampling"

    def test_build_without_hint(self):
        m = build_gsd_estimator_map(train_hint=False)
        assert "gsd_hint" not in m
        assert "gsd_distill_onpolicy" in m

    def test_build_with_hint(self):
        m = build_gsd_estimator_map(train_hint=True)
        assert "gsd_hint" in m


# ---------------------------------------------------------------------------
# build_topk_fkl_datum — shape and weight correctness
# ---------------------------------------------------------------------------


class TestBuildTopkFklDatum:
    @pytest.fixture(autouse=True)
    def _skip_without_tinker(self):
        pytest.importorskip("tinker")

    def test_datum_shape_and_mask(self):
        """Datum has (N, K) targets/weights and (N,) mask via from_torch()."""

        from rllm.experimental.gsd.losses import build_topk_fkl_datum

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

        # After right-shift: N = len(prompt + response) - 1 = 4
        N = len(prompt_ids) + len(response_ids) - 1
        assert target_t.shape == (N, K)
        assert weights_t.shape == (N, K)
        assert mask_t.shape == (N,)

        # Prompt portion: zeros and mask=0
        prompt_len = N - len(response_ids)
        for t in range(prompt_len):
            assert target_t[t].sum().item() == 0
            assert weights_t[t].sum().item() == 0.0
            assert mask_t[t].item() == 0.0

        # Response portion: teacher IDs and renormalized probs, mask=1
        resp_start = prompt_len
        assert target_t[resp_start].tolist() == [10, 20, 30]
        assert mask_t[resp_start].item() == 1.0

        probs = weights_t[resp_start]
        np.testing.assert_allclose(probs.sum().item(), 1.0, atol=1e-6)

    def test_weight_clamp(self):
        """loss_clamp caps the maximum weight per position."""
        from rllm.experimental.gsd.losses import build_topk_fkl_datum

        prompt_ids = [1]
        response_ids = [2]
        teacher_topk = {
            "topk_ids": [[10, 20]],
            "topk_logprobs": [[math.log(0.99), math.log(0.01)]],
        }
        datum = build_topk_fkl_datum(
            prompt_ids,
            response_ids,
            teacher_topk,
            loss_clamp=0.5,
        )
        weights_t = datum.loss_fn_inputs["weights"].to_torch()
        assert weights_t.max().item() <= 0.5 + 1e-8
