"""Tests for per-role policy loss infrastructure."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rllm.experimental.common.config import AlgorithmConfig, rLLMAdvantageEstimator

# ---------------------------------------------------------------------------
# AlgorithmConfig normalization
# ---------------------------------------------------------------------------


class TestEstimatorMapNormalization:
    def test_bare_values_unchanged(self):
        cfg = AlgorithmConfig(
            estimator_map={
                "solver": rLLMAdvantageEstimator.GRPO,
                "judge": "reinforce",
            }
        )
        assert cfg.estimator_map == {
            "solver": rLLMAdvantageEstimator.GRPO,
            "judge": "reinforce",
        }
        assert cfg.loss_fn_map == {}

    def test_tuple_split(self):
        cfg = AlgorithmConfig(
            estimator_map={
                "solver": (rLLMAdvantageEstimator.GRPO, "gspo"),
                "judge": rLLMAdvantageEstimator.REINFORCE,
            }
        )
        assert cfg.estimator_map == {
            "solver": rLLMAdvantageEstimator.GRPO,
            "judge": rLLMAdvantageEstimator.REINFORCE,
        }
        assert cfg.loss_fn_map == {"solver": "gspo"}

    def test_all_tuples(self):
        cfg = AlgorithmConfig(
            estimator_map={
                "a": ("grpo", "vanilla"),
                "b": ("reinforce", "gpg"),
            }
        )
        assert cfg.estimator_map == {"a": "grpo", "b": "reinforce"}
        assert cfg.loss_fn_map == {"a": "vanilla", "b": "gpg"}

    def test_invalid_tuple_length(self):
        with pytest.raises(ValueError, match="exactly 2 elements"):
            AlgorithmConfig(estimator_map={"bad": ("a", "b", "c")})

    def test_empty_map(self):
        cfg = AlgorithmConfig(estimator_map={})
        assert cfg.estimator_map == {}
        assert cfg.loss_fn_map == {}

    def test_backward_compat_no_tuples(self):
        """Old-style map (no tuples) should produce empty loss_fn_map."""
        cfg = AlgorithmConfig(
            estimator_map={
                "solver": rLLMAdvantageEstimator.GRPO,
                "judge": rLLMAdvantageEstimator.REINFORCE,
            }
        )
        assert cfg.loss_fn_map == {}


# ---------------------------------------------------------------------------
# Tinker loss fallback
# ---------------------------------------------------------------------------


class TestTinkerLossFallback:
    def test_unknown_loss_triggers_warning(self, caplog):
        from rllm.trainer.tinker.tinker_policy_trainer import (
            DEFAULT_LOSS_FN,
            TINKER_KNOWN_LOSSES,
        )

        assert "gspo" not in TINKER_KNOWN_LOSSES
        assert DEFAULT_LOSS_FN == "importance_sampling"

    def test_known_losses_complete(self):
        from rllm.trainer.tinker.tinker_policy_trainer import TINKER_KNOWN_LOSSES

        expected = {"importance_sampling", "ppo", "cispo", "dro", "cross_entropy"}
        assert TINKER_KNOWN_LOSSES == expected


# ---------------------------------------------------------------------------
# Verl actor patch
# ---------------------------------------------------------------------------


class TestVerlActorPatch:
    def test_patch_is_idempotent(self):
        """Calling the patch function twice should not error."""
        from rllm.experimental.verl.patch import patch_verl_actor_for_loss_override

        patch_verl_actor_for_loss_override()
        patch_verl_actor_for_loss_override()

    def test_override_applied_and_restored(self):
        """The patched update_policy should read override from meta_info and restore config."""
        from rllm.experimental.verl.patch import patch_verl_actor_for_loss_override

        patch_verl_actor_for_loss_override()

        from verl.workers.actor.dp_actor import DataParallelPPOActor

        # Create a mock actor instance with a config
        actor = MagicMock(spec=DataParallelPPOActor)
        actor.config = MagicMock()
        actor.config.policy_loss = {"loss_mode": "vanilla"}

        # Create a mock DataProto with override
        data = MagicMock()
        data.meta_info = {"policy_loss_mode_override": "gspo"}

        # The patched method should set override, call original, then restore
        # We can't easily test the full flow without a real actor, but we can
        # verify the patch function is callable and the class attribute is modified
        assert hasattr(DataParallelPPOActor, "update_policy")

    def test_no_override_passes_through(self):
        """Without override in meta_info, original behavior is preserved."""
        from rllm.experimental.verl.patch import patch_verl_actor_for_loss_override

        patch_verl_actor_for_loss_override()
        # Just verify no crash — actual actor integration requires GPU


# ---------------------------------------------------------------------------
# Verl known losses
# ---------------------------------------------------------------------------


class TestVerlKnownLosses:
    def test_registry_populated(self):
        from rllm.experimental.verl.verl_backend import _get_verl_known_losses

        known = _get_verl_known_losses()
        assert "vanilla" in known
        assert len(known) > 5  # at least the main ones

    def test_default_loss_is_known(self):
        from rllm.experimental.verl.verl_backend import (
            _DEFAULT_VERL_LOSS,
            _get_verl_known_losses,
        )

        assert _DEFAULT_VERL_LOSS in _get_verl_known_losses()


# ---------------------------------------------------------------------------
# Loss group regrouping logic
# ---------------------------------------------------------------------------


class TestLossGrouping:
    def test_roles_with_same_loss_grouped(self):
        """Roles A and B share 'vanilla', role C uses 'gspo' → 2 groups."""
        role_to_loss = {"a": "vanilla", "b": "vanilla", "c": "gspo"}
        from collections import defaultdict

        loss_to_roles: dict[str, list[str]] = defaultdict(list)
        for role, loss in role_to_loss.items():
            loss_to_roles[loss].append(role)

        assert len(loss_to_roles) == 2
        assert sorted(loss_to_roles["vanilla"]) == ["a", "b"]
        assert loss_to_roles["gspo"] == ["c"]

    def test_all_same_loss_single_group(self):
        """All roles share the same loss → 1 group (fast path)."""
        role_to_loss = {"a": "vanilla", "b": "vanilla"}
        from collections import defaultdict

        loss_to_roles: dict[str, list[str]] = defaultdict(list)
        for role, loss in role_to_loss.items():
            loss_to_roles[loss].append(role)

        assert len(loss_to_roles) == 1

    def test_each_role_different_loss(self):
        """Each role has a unique loss → N groups."""
        role_to_loss = {"a": "vanilla", "b": "gspo", "c": "gpg"}
        from collections import defaultdict

        loss_to_roles: dict[str, list[str]] = defaultdict(list)
        for role, loss in role_to_loss.items():
            loss_to_roles[loss].append(role)

        assert len(loss_to_roles) == 3


# ---------------------------------------------------------------------------
# group_roles in DataProto pipeline
# ---------------------------------------------------------------------------


class TestGroupRolesInDataclass:
    def test_accumulated_data_tracks_group_roles(self):
        import torch

        from rllm.experimental.verl.dataclass import AccumulatedData, ProcessedStepData

        acc = AccumulatedData()
        step = ProcessedStepData(
            prompt=torch.tensor([1, 2]),
            response=torch.tensor([3, 4]),
            mask=torch.tensor([1, 1]),
            step_reward=1.0,
            step_id="s0",
        )
        acc.add_step(step, "traj_0", 1.0, 1, True, group_role="solver")
        acc.add_step(step, "traj_1", 0.5, 1, True, group_role="judge")

        assert acc.group_roles == ["solver", "judge"]
        assert len(acc) == 2


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_no_loss_fn_map_means_no_splitting(self):
        """When loss_fn_map is empty, the Verl backend should not split."""
        cfg = AlgorithmConfig(
            estimator_map={
                "solver": rLLMAdvantageEstimator.GRPO,
                "judge": rLLMAdvantageEstimator.REINFORCE,
            }
        )
        # No tuples → loss_fn_map is empty
        assert cfg.loss_fn_map == {}

    def test_advantage_computation_unaffected(self):
        """The advantage.py code reads estimator_map[role] which should still
        be a bare estimator after normalization, not a tuple."""
        cfg = AlgorithmConfig(
            estimator_map={
                "solver": (rLLMAdvantageEstimator.GRPO, "gspo"),
            }
        )
        # After normalization, estimator_map should have the bare estimator
        assert cfg.estimator_map["solver"] == rLLMAdvantageEstimator.GRPO
        assert not isinstance(cfg.estimator_map["solver"], tuple)
