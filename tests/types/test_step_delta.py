"""The completed-parent StepDelta contract."""

import json

import pytest
from pydantic import ValidationError
from rllm_model_gateway.models import _messages_start_with

from rllm.engine.rollout import ModelOutput
from rllm.types import Step, StepDelta, resolve_step_deltas


def _step(
    step_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    chat_completions: list[dict],
    *,
    lineage: str | None = "main",
) -> Step:
    response = chat_completions[-1] if chat_completions else {}
    content = response.get("content", "") or ""
    reasoning = response.get("reasoning", "") or ""
    logprobs = [-0.1] * len(response_ids)
    return Step(
        id=step_id,
        input={"step": step_id},
        output={"answer": step_id},
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        logprobs=logprobs,
        chat_completions=chat_completions,
        thought=reasoning,
        model_response=content,
        model_output=ModelOutput(
            content=content,
            reasoning=reasoning,
            prompt_ids=prompt_ids,
            completion_ids=response_ids,
            logprobs=logprobs,
            prompt_length=len(prompt_ids),
            completion_length=len(response_ids),
        ),
        reward=1.0,
        metadata=None if lineage is None else {"lineage_id": lineage},
    )


def _cumulative(turns: int, *, lineage: str = "main") -> list[Step]:
    """Production shape: each Step includes its current assistant response."""
    request_messages = [{"role": "system", "content": "system"}]
    prompt_ids = [1, 2]
    steps = []
    for turn in range(turns):
        request_messages = [
            *request_messages,
            {"role": "user", "content": f"question {turn}"},
        ]
        prompt_ids = [*prompt_ids, 10 + turn]
        response = {"role": "assistant", "content": f"answer {turn}"}
        response_ids = [100 + turn, 200 + turn]
        steps.append(
            _step(
                f"{lineage}.s{turn}",
                prompt_ids,
                response_ids,
                [*request_messages, response],
                lineage=lineage,
            )
        )
        request_messages = [*request_messages, response]
        prompt_ids = [*prompt_ids, *response_ids]
    return steps


def _encode_against(step: Step, parent: Step | None) -> StepDelta:
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
    return StepDelta(
        id=step.id,
        parent_step_id=None if parent is None else parent.id,
        input=step.input,
        output=step.output,
        action=step.action,
        reward=step.reward,
        done=step.done,
        metadata=step.metadata,
        prompt_ids_suffix=step.prompt_ids[len(prompt_prefix) :],
        chat_completions_suffix=step.chat_completions[len(chat_prefix) :],
        response_ids=step.response_ids,
        logprobs=step.logprobs,
        routing_matrices=step.routing_matrices,
        observation=step.observation,
        thought=step.thought,
        model_response=step.model_response,
        finish_reason=getattr(step.model_output, "finish_reason", None),
        mc_return=step.mc_return,
        advantage=step.advantage,
        weight_version=step.weight_version,
    )


def _deltas(steps: list[Step]) -> list[StepDelta]:
    return [_encode_against(step, None if index == 0 else steps[index - 1]) for index, step in enumerate(steps)]


def _bytes(step: Step) -> bytes:
    return json.dumps(
        step.model_dump(mode="json"),
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def test_completed_parent_roundtrip_is_exact():
    steps = _cumulative(8)
    assert [_bytes(step) for step in resolve_step_deltas(_deltas(steps))] == [_bytes(step) for step in steps]


def test_delta_stores_only_new_request_and_response_content():
    steps = _cumulative(8)
    deltas = _deltas(steps)
    assert deltas[0].parent_step_id is None
    assert deltas[0].prompt_ids_suffix == steps[0].prompt_ids
    for index, delta in enumerate(deltas[1:], 1):
        assert delta.parent_step_id == steps[index - 1].id
        assert len(delta.prompt_ids_suffix) == 1
        assert len(delta.chat_completions_suffix) == 2


def test_parent_completion_tokens_are_inherited_not_repeated():
    parent, child = _cumulative(2)
    delta = _encode_against(child, parent)
    prefix = [*parent.prompt_ids, *parent.response_ids]
    assert child.prompt_ids == [
        *prefix,
        *delta.prompt_ids_suffix,
    ]
    assert delta.prompt_ids_suffix == child.prompt_ids[len(prefix) :]


def test_history_rewrite_becomes_self_contained_root():
    parent = _cumulative(1)[0]
    rewritten = _step(
        "rewrite",
        [9, 8],
        [7],
        [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "ok"}],
    )
    delta = _encode_against(rewritten, parent)
    assert delta.parent_step_id is None
    assert _bytes(resolve_step_deltas([delta])[0]) == _bytes(rewritten)


def test_message_key_order_mismatch_becomes_root_for_byte_parity():
    parent = _step(
        "parent",
        [1],
        [],
        [{"role": "assistant", "content": "x"}],
    )
    child = _step(
        "child",
        [1, 2],
        [],
        [
            {"content": "x", "role": "assistant"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "y"},
        ],
    )
    delta = _encode_against(child, parent)
    assert delta.parent_step_id is None
    assert list(resolve_step_deltas([delta])[0].chat_completions[0]) == [
        "content",
        "role",
    ]


def test_same_lineage_siblings_resolve_the_named_parent():
    root = _cumulative(1)[0]
    left = _step(
        "left",
        [*root.prompt_ids, *root.response_ids, 31],
        [41],
        [
            *root.chat_completions,
            {"role": "user", "content": "left"},
            {"role": "assistant", "content": "L"},
        ],
    )
    right = _step(
        "right",
        [*root.prompt_ids, *root.response_ids, 32],
        [42],
        [
            *root.chat_completions,
            {"role": "user", "content": "right"},
            {"role": "assistant", "content": "R"},
        ],
    )
    deltas = [
        _encode_against(root, None),
        _encode_against(left, root),
        _encode_against(right, root),
    ]
    assert deltas[1].parent_step_id == deltas[2].parent_step_id == root.id
    assert [_bytes(step) for step in resolve_step_deltas(deltas)] == [
        _bytes(root),
        _bytes(left),
        _bytes(right),
    ]


def test_interleaved_lineages_resolve_by_id():
    a = _cumulative(2, lineage="a")
    b = _cumulative(2, lineage="b")
    deltas = [
        _encode_against(a[0], None),
        _encode_against(b[0], None),
        _encode_against(a[1], a[0]),
        _encode_against(b[1], b[0]),
    ]
    expected = [a[0], b[0], a[1], b[1]]
    assert [_bytes(step) for step in resolve_step_deltas(deltas)] == [_bytes(step) for step in expected]


def test_input_output_and_none_metadata_survive_json_wire():
    step = _cumulative(1)[0]
    step.metadata = None
    delta = _encode_against(step, None)
    revived = StepDelta.model_validate_json(delta.model_dump_json())
    resolved = resolve_step_deltas([revived])[0]
    assert resolved.input == step.input
    assert resolved.output == step.output
    assert resolved.metadata is None
    assert _bytes(resolved) == _bytes(step)


def test_model_output_is_reconstructed_without_storing_its_prompt():
    step = _cumulative(1)[0]
    content = step.chat_completions[-1]["content"]
    step.output = step.model_response = content
    step.model_output = ModelOutput(
        content=content,
        reasoning="",
        prompt_ids=step.prompt_ids,
        completion_ids=step.response_ids,
        logprobs=step.logprobs,
        prompt_length=len(step.prompt_ids),
        completion_length=len(step.response_ids),
        finish_reason="stop",
    )
    delta = _encode_against(step, None)
    assert not hasattr(delta, "model_output")
    assert _bytes(resolve_step_deltas([delta])[0]) == _bytes(step)


def test_missing_token_ids_are_valid_and_lossless():
    parent = _step(
        "parent",
        [],
        [],
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    )
    child = _step(
        "child",
        [],
        [],
        [
            *parent.chat_completions,
            {"role": "user", "content": "z"},
            {"role": "assistant", "content": "w"},
        ],
    )
    deltas = [_encode_against(parent, None), _encode_against(child, parent)]
    assert [_bytes(step) for step in resolve_step_deltas(deltas)] == [
        _bytes(parent),
        _bytes(child),
    ]


def test_resolved_steps_share_prefix_message_objects():
    resolved = resolve_step_deltas(_deltas(_cumulative(4)))
    assert resolved[0].chat_completions[0] is resolved[-1].chat_completions[0]


def test_resolved_values_do_not_mutate_delta_storage():
    delta = _encode_against(_cumulative(1)[0], None)
    delta.routing_matrices = ["route"]
    resolved = resolve_step_deltas([delta])[0]
    resolved.chat_completions[0]["content"] = "changed"
    resolved.response_ids.append(999)
    resolved.model_output.completion_ids.append(998)
    resolved.model_output.logprobs.append(-1.0)
    resolved.model_output.routing_matrices.append("changed")
    resolved.metadata["changed"] = True
    again = resolve_step_deltas([delta])[0]
    assert again.chat_completions[0]["content"] == "system"
    assert 999 not in again.response_ids
    assert 998 not in again.response_ids
    assert -1.0 not in again.logprobs
    assert again.routing_matrices == ["route"]
    assert "changed" not in again.metadata


def test_storage_is_linear_for_cumulative_trajectory():
    steps = _cumulative(100)
    deltas = _deltas(steps)
    stored_messages = sum(len(delta.chat_completions_suffix) for delta in deltas)
    stored_prompt_ids = sum(len(delta.prompt_ids_suffix) for delta in deltas)
    assert stored_messages == len(steps[-1].chat_completions)
    assert stored_prompt_ids == len(steps[-1].prompt_ids) - sum(len(step.response_ids) for step in steps[:-1])
    assert sum(len(step.chat_completions) for step in steps) > 40 * stored_messages


def test_duplicate_and_dangling_parent_ids_are_rejected():
    root = _encode_against(_cumulative(1)[0], None)
    duplicate = root.model_copy()
    with pytest.raises(ValueError, match="duplicate step id"):
        resolve_step_deltas([root, duplicate])

    dangling = root.model_copy(update={"id": "child", "parent_step_id": "missing"})
    with pytest.raises(ValueError, match="not earlier"):
        resolve_step_deltas([dangling])


def test_parent_and_suffix_fields_are_required():
    with pytest.raises(ValidationError):
        StepDelta(id="missing")
