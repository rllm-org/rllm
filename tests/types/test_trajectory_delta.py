"""TrajectoryDelta is the StepDelta container and parity oracle."""

import json

import pytest
from rllm_model_gateway.models import _messages_start_with

from rllm.engine.rollout import ModelOutput
from rllm.types import Step, StepDelta, Trajectory, TrajectoryDelta


def _steps(turns: int, lineage: str = "main", base: int = 0) -> list[Step]:
    messages = [{"role": "system", "content": f"system {base}"}]
    prompt_ids = [base + 1]
    steps = []
    for turn in range(turns):
        messages = [*messages, {"role": "user", "content": f"question {turn}"}]
        prompt_ids = [*prompt_ids, base + 10 + turn]
        response = {"role": "assistant", "content": f"answer {turn}"}
        response_ids = [base + 100 + turn]
        steps.append(
            Step(
                id=f"{lineage}.{turn}",
                input={"turn": turn},
                output={"answer": turn},
                action={"tool": turn},
                reward=float(turn),
                done=turn == turns - 1,
                metadata={"lineage_id": lineage, "turn": turn},
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                logprobs=[-0.1],
                routing_matrices=["header", "payload"],
                chat_completions=[*messages, response],
                observation={"seen": turn},
                thought=f"thought {turn}",
                model_response=f"answer {turn}",
                model_output=ModelOutput(
                    content=f"answer {turn}",
                    reasoning="",
                    prompt_ids=prompt_ids,
                    completion_ids=response_ids,
                    logprobs=[-0.1],
                    routing_matrices=["header", "payload"],
                    prompt_length=len(prompt_ids),
                    completion_length=len(response_ids),
                    finish_reason="stop",
                    weight_version=7,
                ),
                mc_return=0.5,
                advantage=[0.25],
                weight_version=7,
            )
        )
        messages = [*messages, response]
        prompt_ids = [*prompt_ids, *response_ids]
    return steps


def _bytes(trajectory: Trajectory) -> bytes:
    return json.dumps(trajectory.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _trajectory(steps: list[Step]) -> Trajectory:
    return Trajectory(
        uid="trajectory",
        name="agent",
        task={"question": "unicode ✓"},
        steps=steps,
        reward=1.0,
        input={"argument": 1},
        output={"result": True},
        signals={"score": 1.0},
        metadata={"source": "test"},
    )


def _encode_step(step: Step, parent: Step | None) -> StepDelta:
    """Test-only flat encoder; production receives explicit TraceDelta edges."""
    chat_prefix = [] if parent is None else parent.chat_completions
    prompt_prefix = [] if parent is None else [*parent.prompt_ids, *parent.response_ids]
    child = (
        parent is not None
        and (parent.metadata or {}).get("lineage_id") == (step.metadata or {}).get("lineage_id")
        and _messages_start_with(step.chat_completions, chat_prefix)
        and step.prompt_ids[: len(prompt_prefix)] == prompt_prefix
    )
    if not child:
        parent = None
        chat_prefix, prompt_prefix = [], []
    step_values = {name: getattr(step, name) for name in StepDelta.model_fields if name in Step.model_fields}
    return StepDelta(
        **step_values,
        parent_step_id=None if parent is None else parent.id,
        prompt_ids_suffix=step.prompt_ids[len(prompt_prefix) :],
        chat_completions_suffix=step.chat_completions[len(chat_prefix) :],
        finish_reason=getattr(step.model_output, "finish_reason", None),
    )


def _encode_trajectory(trajectory: Trajectory) -> TrajectoryDelta:
    last: dict[str | None, Step] = {}
    deltas = []
    for step in trajectory.steps:
        lineage = (step.metadata or {}).get("lineage_id")
        deltas.append(_encode_step(step, last.get(lineage)))
        last[lineage] = step
    fields = {name: getattr(trajectory, name) for name in Trajectory.model_fields if name != "steps"}
    return TrajectoryDelta(**fields, steps=deltas)


def test_gateway_trajectory_json_wire_roundtrip_is_exact():
    assert list(TrajectoryDelta.model_fields) == list(Trajectory.model_fields)
    assert TrajectoryDelta.model_fields["steps"].annotation == list[StepDelta]
    for name, field in Trajectory.model_fields.items():
        if name != "steps":
            mirror = TrajectoryDelta.model_fields[name]
            assert (mirror.annotation, mirror.default, mirror.default_factory is None) == (field.annotation, field.default, field.default_factory is None)

    flat = _trajectory(_steps(4))
    delta = _encode_trajectory(flat)
    revived = TrajectoryDelta.model_validate_json(delta.model_dump_json())

    assert [step.parent_step_id for step in revived.steps] == [None, "main.0", "main.1", "main.2"]
    assert _bytes(revived.resolve()) == _bytes(flat)


def test_interleaved_lineages_resolve_independently():
    left = _steps(2, "left", 100)
    right = _steps(2, "right", 200)
    flat = _trajectory([left[0], right[0], left[1], right[1]])
    delta = _encode_trajectory(flat)

    assert [step.parent_step_id for step in delta.steps] == [None, None, "left.0", "right.0"]
    assert _bytes(delta.resolve()) == _bytes(flat)


def test_resolve_preserves_arbitrary_trajectory_values():
    value = object()
    flat = _trajectory([])
    flat.task = flat.output = value

    resolved = _encode_trajectory(flat).resolve()
    assert resolved.task is value
    assert resolved.output is value


def test_resolve_follows_parent_ids_for_a_branching_dag():
    root = _steps(1)[0]

    def child(step_id: str, token: int) -> Step:
        prompt_ids = [*root.prompt_ids, *root.response_ids, token]
        response_ids = [token + 1]
        content = f"answer {step_id}"
        chat_completions = [
            *root.chat_completions,
            {"role": "user", "content": step_id},
            {"role": "assistant", "content": content},
        ]
        return Step(
            id=step_id,
            metadata={"lineage_id": "main"},
            output=content,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            chat_completions=chat_completions,
            model_response=content,
            model_output=ModelOutput(
                content=content,
                reasoning="",
                prompt_ids=prompt_ids,
                completion_ids=response_ids,
                logprobs=[],
                prompt_length=len(prompt_ids),
                completion_length=len(response_ids),
            ),
        )

    left, right = child("left", 201), child("right", 301)
    delta = TrajectoryDelta(
        **{name: getattr(_trajectory([]), name) for name in Trajectory.model_fields if name != "steps"},
        steps=[
            _encode_step(root, None),
            _encode_step(left, root),
            _encode_step(right, root),
        ],
    )

    assert [step.parent_step_id for step in delta.steps] == [None, root.id, root.id]
    assert _bytes(delta.resolve()) == _bytes(_trajectory([root, left, right]))


def test_resolve_rejects_a_dangling_parent():
    delta = _encode_trajectory(_trajectory(_steps(2)))
    delta.steps[1] = delta.steps[1].model_copy(update={"parent_step_id": "missing"})
    with pytest.raises(ValueError, match="not earlier"):
        delta.resolve()
