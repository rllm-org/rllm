import torch

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.experimental.verl.collapse import collapse_trajectory_steps, is_prefix
from rllm.experimental.verl.dataclass import AccumulatedData
from rllm.experimental.verl.metrics import compute_rollout_probs_diff_metrics


def _make_step(prompt_ids: list[int], completion_ids: list[int], reward: float = 0.0, logprobs: list[float] | None = None) -> Step:
    model_output = ModelOutput(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
    )
    return Step(model_output=model_output, reward=reward)


def _make_trajectory(steps: list[Step], reward: float = 1.0, name: str = "default_traj_name") -> Trajectory:
    return Trajectory(steps=steps, reward=reward, name=name)


def _run_collapse(trajectory: Trajectory, task_id: str = "task0") -> AccumulatedData:
    accumulated = AccumulatedData()
    for collapsed_step in collapse_trajectory_steps(trajectory, task_id):
        accumulated.add_step(
            step_data=collapsed_step.step_data,
            trajectory_id=collapsed_step.trajectory_id,
            traj_reward=collapsed_step.traj_reward,
            step_num=collapsed_step.step_num,
            is_last=collapsed_step.is_last_step,
            group_role=collapsed_step.group_role,
        )
    return accumulated


class TestIsPrefix:
    def test_exact_prefix(self):
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5])
        step_prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        assert is_prefix(prompt, response, step_prompt) is True

    def test_not_prefix(self):
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5])
        step_prompt = torch.tensor([1, 2, 3, 99, 5, 6, 7])
        assert is_prefix(prompt, response, step_prompt) is False

    def test_prefix_longer_than_step(self):
        prompt = torch.tensor([1, 2, 3])
        response = torch.tensor([4, 5, 6, 7, 8])
        step_prompt = torch.tensor([1, 2, 3, 4])
        assert is_prefix(prompt, response, step_prompt) is False


class TestCollapseTrajectory:
    def test_two_cumulative_steps_collapse(self):
        trajectory = _make_trajectory(
            [
                _make_step([1, 2], [3, 4], reward=0.0),
                _make_step([1, 2, 3, 4, 5, 6], [7, 8], reward=1.0),
            ]
        )
        accumulated = _run_collapse(trajectory)

        assert len(accumulated) == 1
        assert accumulated.prompts[0].tolist() == [1, 2]
        assert accumulated.responses[0].tolist() == [3, 4, 5, 6, 7, 8]
        assert accumulated.traj_mask[0].tolist() == [1, 1, 0, 0, 1, 1]
        assert accumulated.is_last_step[0] is True

    def test_prefix_break_creates_two_sequences(self):
        trajectory = _make_trajectory(
            [
                _make_step([1, 2], [3, 4], reward=0.0),
                _make_step([1, 2, 3, 4, 5], [6, 7], reward=0.0),
                _make_step([99, 100], [101], reward=1.0),
            ]
        )
        accumulated = _run_collapse(trajectory)

        assert len(accumulated) == 2
        assert accumulated.responses[0].tolist() == [3, 4, 5, 6, 7]
        assert accumulated.traj_mask[0].tolist() == [1, 1, 0, 1, 1]
        assert accumulated.is_last_step[0] is False
        assert accumulated.responses[1].tolist() == [101]
        assert accumulated.is_last_step[1] is True

    def test_logprobs_preserved_after_merge(self):
        trajectory = _make_trajectory(
            [
                _make_step([1, 2], [3, 4], reward=0.0, logprobs=[-0.1, -0.2]),
                _make_step([1, 2, 3, 4, 5], [6], reward=1.0, logprobs=[-0.3]),
            ]
        )
        accumulated = _run_collapse(trajectory)

        assert len(accumulated.rollout_logprobs) == 1
        assert torch.allclose(
            accumulated.rollout_logprobs[0],
            torch.tensor([-0.1, -0.2, 0.0, -0.3], dtype=torch.float32),
        )

    def test_merge_uses_latest_multimodal_inputs(self):
        first = Step(
            model_output=ModelOutput(
                prompt_ids=[1, 2],
                completion_ids=[3, 4],
                multi_modal_inputs={"image_grid_thw": torch.tensor([[1, 8, 8]])},
                prompt_length=2,
                completion_length=2,
            ),
            reward=0.0,
        )
        second = Step(
            model_output=ModelOutput(
                prompt_ids=[1, 2, 3, 4, 5],
                completion_ids=[6],
                multi_modal_inputs={"image_grid_thw": torch.tensor([[1, 8, 8], [1, 4, 4]])},
                prompt_length=5,
                completion_length=1,
            ),
            reward=1.0,
        )
        accumulated = _run_collapse(_make_trajectory([first, second]))

        assert accumulated.multi_modal_inputs[0]["image_grid_thw"].shape == (2, 3)


class TestRolloutProbDiffMetrics:
    def test_gap_tokens_are_excluded_from_rollout_diff(self):
        rollout_old_log_probs = torch.log(torch.tensor([[0.8, 0.9, 0.2, 0.7]], dtype=torch.float32))
        actor_old_log_probs = torch.log(torch.tensor([[0.7, 0.85, 0.95, 0.6]], dtype=torch.float32))
        response_mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.long)

        metrics = compute_rollout_probs_diff_metrics(rollout_old_log_probs, actor_old_log_probs, response_mask)
        expected = torch.tensor([0.1, 0.05, 0.1], dtype=torch.float32)

        assert torch.isclose(torch.tensor(metrics["training/rollout_probs_diff_max"]), torch.max(expected))
        assert torch.isclose(torch.tensor(metrics["training/rollout_probs_diff_mean"]), torch.mean(expected))
        assert torch.isclose(torch.tensor(metrics["training/rollout_probs_diff_std"]), torch.std(expected))
