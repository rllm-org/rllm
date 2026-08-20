from rllm.eval.visualizer import _episode_payload_for_view
from rllm.types import Episode, StepDelta, TrajectoryDelta


def test_compact_episode_is_materialized_for_view():
    root = StepDelta(
        id="root",
        parent_step_id=None,
        prompt_ids_suffix=[1],
        chat_completions_suffix=[
            {"role": "user", "content": "Q0"},
            {"role": "assistant", "content": "A0"},
        ],
        response_ids=[2],
        model_response="A0",
    )
    child = StepDelta(
        id="child",
        parent_step_id="root",
        prompt_ids_suffix=[3],
        chat_completions_suffix=[
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ],
        response_ids=[4],
        model_response="A1",
    )
    payload = Episode(id="task:0", trajectories=[TrajectoryDelta(steps=[root, child])]).model_dump(mode="json")
    payload["eval_idx"] = 7

    viewed = _episode_payload_for_view(payload)
    first, second = viewed["trajectories"][0]["steps"]

    assert viewed["eval_idx"] == 7
    assert first["prompt_ids"] == [1]
    assert second["prompt_ids"] == [1, 2, 3]
    assert second["chat_completions"] == [*root.chat_completions_suffix, *child.chat_completions_suffix]
    assert "prompt_ids_suffix" not in second


def test_full_episode_payload_is_unchanged():
    payload = {"id": "full", "trajectories": [{"steps": [{"prompt_ids": [1]}]}]}
    assert _episode_payload_for_view(payload) is payload
