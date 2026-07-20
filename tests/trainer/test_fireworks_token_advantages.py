import pytest
from training.utils.rl.losses import build_builtin_loss_datums

from rllm.trainer.fireworks.fireworks_policy_trainer import FireworksPolicyTrainer
from rllm.trainer.tinker.transform import trajectory_to_datums
from rllm.types import Step, Trajectory


@pytest.mark.parametrize(
    ("advantage", "expected"),
    [([0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]), (0.5, [0.0, 0.5, 0.5, 0.5])],
)
def test_native_fireworks_path_preserves_scalar_or_token_advantages(advantage, expected):
    trajectory = Trajectory(
        steps=[
            Step(
                prompt_ids=[1, 2],
                response_ids=[3, 4, 5],
                logprobs=[-0.1, -0.2, -0.3],
                advantage=advantage,
            )
        ]
    )
    raw_datums = trajectory_to_datums(trajectory)
    clean, advantages, inf_logprobs, prompt_lens, _ = FireworksPolicyTrainer._process_datums(raw_datums)

    native_datums = build_builtin_loss_datums(
        clean,
        [1.0],
        inf_logprobs,
        inf_logprobs,
        prompt_lens,
    )
    native_datums = FireworksPolicyTrainer._apply_token_advantages(native_datums, advantages)

    assert native_datums[0].loss_fn_inputs["advantages"].data == pytest.approx(expected)
