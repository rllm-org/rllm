"""Miles backend checks that need a real Miles checkout or install.

Skipped when neither is present, so the suite still runs on a plain rLLM env.
Point ``MILES_ROOT`` at a Miles checkout (default ``~/miles``) to enable the
source audit; the conversion test additionally needs ``miles`` importable, which
means ``ray`` and a Miles checkout on ``sys.path``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

MILES_ROOT = Path(os.environ.get("MILES_ROOT", Path.home() / "miles"))

needs_checkout = pytest.mark.skipif(
    not (MILES_ROOT / "miles" / "utils" / "arguments.py").exists(),
    reason=f"no Miles checkout at {MILES_ROOT} (set MILES_ROOT)",
)


def _miles_importable() -> bool:
    if importlib.util.find_spec("miles") is not None:
        return True
    if not (MILES_ROOT / "miles" / "__init__.py").exists():
        return False
    import sys

    if str(MILES_ROOT) not in sys.path:
        sys.path.insert(0, str(MILES_ROOT))
    try:
        importlib.import_module("miles.ray.rollout.train_data_conversion")
        return True
    except Exception:
        return False


needs_miles = pytest.mark.skipif(not _miles_importable(), reason="miles not importable (needs ray + a Miles checkout)")


def _miles_arguments_importable() -> bool:
    """miles.utils.arguments additionally needs sglang (ServerArgs.add_cli_args)."""
    if not _miles_importable():
        return False
    try:
        importlib.import_module("miles.utils.arguments")
        return True
    except Exception:
        return False


needs_miles_arguments = pytest.mark.skipif(not _miles_arguments_importable(), reason="miles.utils.arguments not importable (needs sglang)")


def _rendered_argv_block():
    from omegaconf import OmegaConf

    import rllm.trainer.config as cfg_pkg
    from rllm.trainer.miles.miles_config import build_block

    cfg = OmegaConf.load(Path(cfg_pkg.__file__).parent / "rllm" / "backend" / "miles.yaml")
    for path, value in [
        ("training.group_size", 8),
        ("rllm.data.train_batch_size", 16),
        ("rllm.data.max_prompt_length", 4096),
        ("rllm.trainer.save_freq", 20),
        ("rllm.trainer.test_freq", 5),
        ("rllm.trainer.project_name", "rllm-miles-test"),
        ("rllm.trainer.experiment_name", "t"),
    ]:
        OmegaConf.update(cfg, path, value)
    return build_block(cfg, total_steps=50)


@needs_checkout
class TestFlagSurfaceMatchesMiles:
    """Catches flag renames/removals in Miles without needing to import it."""

    def test_every_emitted_flag_exists_in_miles(self):
        from rllm.trainer.miles._flag_audit import flag_arity_from_source

        flags = flag_arity_from_source(MILES_ROOT)
        assert len(flags) > 200, f"only found {len(flags)} flags; the audit probably failed to parse"

        block, _ = _rendered_argv_block()
        unknown = sorted(k for k, v in block.items() if v is not None and k not in flags)
        assert not unknown, f"flags the bridge emits that Miles does not define: {unknown}"

    def test_assumed_arity_matches_miles(self):
        from rllm.trainer.miles._flag_audit import flag_arity_from_source
        from rllm.trainer.miles.miles_config import _infer_arity

        flags = flag_arity_from_source(MILES_ROOT)
        block, _ = _rendered_argv_block()
        mismatched = {key: (_infer_arity(value), flags[key]) for key, value in block.items() if value is not None and key in flags and _infer_arity(value) != flags[key]}
        assert not mismatched, f"arity disagreements (assumed, actual): {mismatched}"

    def test_pinned_flags_all_exist(self):
        from rllm.trainer.miles._flag_audit import flag_arity_from_source
        from rllm.trainer.miles.miles_config import PINNED_FLAGS

        flags = flag_arity_from_source(MILES_ROOT)
        missing = sorted(f for f in PINNED_FLAGS if f not in flags)
        assert not missing, f"pinned flags missing from Miles: {missing}"


@needs_miles
class TestSamplesFeedMilesConverter:
    """The transform's output must satisfy Miles' own converter and its assertions."""

    def _train_data(self):
        from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data

        from rllm.trainer.miles.transform import payloads_to_samples, trajectory_groups_to_payloads
        from rllm.types import Step, Trajectory, TrajectoryGroup

        def step(prompt, response, advantage):
            return Step(prompt_ids=list(prompt), response_ids=list(response), logprobs=[-0.5] * len(response), advantage=advantage)

        groups = [
            TrajectoryGroup(
                group_id="p0:agent",
                trajectories=[
                    # merged two-turn chain: observation token 9 sits inside the response
                    Trajectory(steps=[step([1, 2], [3, 4], 1.0), step([1, 2, 3, 4, 9], [5], 1.0)], reward=1.0),
                    Trajectory(steps=[step([1, 2], [6, 7], -1.0)], reward=0.0),
                ],
            ),
            TrajectoryGroup(
                group_id="p1:agent",
                trajectories=[
                    Trajectory(steps=[step([20, 21], [22], 0.5)], reward=1.0),
                    Trajectory(steps=[step([20, 21], [23, 24], -0.5)], reward=0.0),
                ],
            ),
        ]
        samples, advantages = payloads_to_samples(trajectory_groups_to_payloads(groups))

        class Args:
            advantage_estimator = "grpo"
            rewards_normalization = True
            grpo_std_normalization = True
            use_dynamic_global_batch_size = False
            n_samples_per_prompt = 2
            rollout_batch_size = 2
            reward_key = None

        train_data = convert_samples_to_train_data(Args(), samples, metadata={}, custom_convert_samples_to_train_data_func=None, custom_reward_post_process_func=None)
        train_data["advantages"] = advantages
        return train_data

    def test_converter_accepts_our_samples(self):
        data = self._train_data()
        assert data["response_lengths"] == [4, 2, 1, 2]

    def test_every_per_token_array_equals_response_length(self):
        # This is the assertion slice_log_prob_with_cp makes on the trainer side; if it
        # fails there it fails inside a Megatron forward pass, which is a miserable place
        # to debug. Catch it here instead.
        data = self._train_data()
        for i, n in enumerate(data["response_lengths"]):
            assert len(data["loss_masks"][i]) == n, f"row {i} loss_mask"
            assert len(data["rollout_log_probs"][i]) == n, f"row {i} rollout_log_probs"
            assert len(data["advantages"][i]) == n, f"row {i} advantages"

    def test_observation_tokens_are_excluded_from_the_mask_total(self):
        # Row 0 has 4 response tokens but only 3 action tokens (9 is an observation).
        data = self._train_data()
        assert data["rollout_mask_sums"][0] == 3
        assert data["loss_masks"][0] == [1, 1, 0, 1]

    def test_group_index_drives_grpo_reward_grouping(self):
        # Two groups of two, each normalized within itself -> symmetric pairs.
        data = self._train_data()
        rewards = data["rewards"]
        assert rewards[0] == pytest.approx(-rewards[1])
        assert rewards[2] == pytest.approx(-rewards[3])
        assert rewards[0] > 0 and rewards[1] < 0

    def test_raw_rewards_survive_normalization(self):
        assert self._train_data()["raw_reward"] == [1.0, 0.0, 1.0, 0.0]


@needs_miles_arguments
class TestConfigBridgeThroughMilesParser:
    """The bridge's argv must survive Miles' own parse_args and validators.

    These are the checks the static flag audit cannot make: Miles' validators reject
    combinations (eval_interval without eval datasets, save_interval without --save),
    and resolve_rollout_function_paths can *overwrite* a pinned rollout function.
    """

    def _args(self):
        from omegaconf import OmegaConf

        import rllm.trainer.config as cfg_pkg
        from rllm.trainer.miles.miles_config import build_miles_args

        cfg = OmegaConf.load(Path(cfg_pkg.__file__).parent / "rllm" / "backend" / "miles.yaml")
        for path, value in [
            ("training.group_size", 8),
            ("rllm.data.train_batch_size", 16),
            ("rllm.data.max_prompt_length", 4096),
            ("rllm.trainer.save_freq", 20),
            ("rllm.trainer.project_name", "rllm-miles-test"),
            ("rllm.trainer.experiment_name", "t"),
        ]:
            OmegaConf.update(cfg, path, value)
        return build_miles_args(cfg, total_steps=50)

    def test_arity_introspection_finds_the_full_flag_surface(self):
        from rllm.trainer.miles.miles_config import miles_arity

        # ~1500 at the time of writing: Miles' own flags plus the --sglang-* /
        # --eval-sglang-* families generated from ServerArgs.add_cli_args.
        arity = miles_arity(["--rollout-batch-size", "1"])
        assert len(arity) > 800
        assert arity["disable_rollout_global_dataset"] == 0
        assert arity["rollout_function_path"] == 1
        assert arity["ft_components"] == "+"

    def test_rendered_argv_parses(self):
        args = self._args()
        assert args.train_backend == "fsdp"
        assert args.num_rollout == 50
        assert args.rollout_batch_size == 16
        assert args.global_batch_size == 32

    def test_dataset_pin_survives_validation(self):
        assert self._args().rollout_global_dataset is False

    def test_advantage_pin_survives_validation(self):
        # With this False, Miles skips its own estimator and the loss reads the
        # per-token advantages rLLM ships in the batch.
        assert self._args().compute_advantages_and_returns is False

    def test_rollout_function_pin_is_not_overwritten(self):
        # resolve_rollout_function_paths substitutes a standard path when this is
        # unset, and FullyAsyncRolloutFn when --fully-async is on.
        args = self._args()
        assert args.rollout_function_path == "miles.rollout.sleep_rollout.sleep"
        assert args.eval_function_path == "miles.rollout.sleep_rollout.sleep"

    def test_save_dir_is_derived_from_the_run_identity(self):
        # miles_validate_args requires --save whenever --save-interval is set.
        assert self._args().save == "checkpoints/rllm-miles-test/t"

    def test_no_critic_and_default_loss(self):
        args = self._args()
        assert args.use_critic is False
        assert args.loss_type == "policy_loss"  # stock kernel reads batch["advantages"]
