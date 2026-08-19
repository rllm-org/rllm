"""Compact and flat training inputs must produce exactly the same rows."""

import asyncio
import base64
import importlib
import json
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from rllm_model_gateway.models import _messages_start_with

from rllm.engine.rollout import ModelOutput
from rllm.types import Action, Episode, Step, StepDelta, Trajectory, TrajectoryDelta, TrajectoryGroup, _index_step_deltas


def _step(step_id, prompt, response, chat, lineage="main"):
    logprobs = [-0.1] * len(response)
    return Step(
        id=step_id,
        prompt_ids=list(prompt),
        response_ids=list(response),
        logprobs=logprobs,
        chat_completions=chat,
        model_output=ModelOutput(
            content=chat[-1].get("content", ""),
            reasoning="",
            prompt_ids=list(prompt),
            completion_ids=list(response),
            logprobs=logprobs,
            prompt_length=len(prompt),
            completion_length=len(response),
        ),
        advantage=0.5,
        metadata={"lineage_id": lineage},
    )


def _chain(prefix, lineage="main"):
    steps, prompt, chat = [], [prefix], []
    for i in range(3):
        chat = [*chat, {"role": "user", "content": f"u{i}"}]
        response = [prefix + 10 + i]
        step_chat = [*chat, {"role": "assistant", "content": f"a{i}"}]
        steps.append(_step(f"{lineage}-{i}", prompt, response, step_chat, lineage))
        prompt = [*prompt, *response, prefix + 20 + i]
        chat = step_chat
    return steps


def _forms(steps, parents=None):
    flat = Trajectory(uid="trajectory", steps=steps, reward=1.0)
    if parents is None:
        last = {}
        parent_steps = []
        for step in steps:
            lineage = (step.metadata or {}).get("lineage_id")
            parent_steps.append(last.get(lineage))
            last[lineage] = step
    else:
        parent_steps = [None if parent is None else steps[parent] for parent in parents]
    deltas = [_encode_step(step, parent) for step, parent in zip(steps, parent_steps, strict=True)]
    return flat, TrajectoryDelta(uid=flat.uid, steps=deltas, reward=flat.reward)


def _encode_step(step, parent):
    """Test-only flat encoder; production maps explicit TraceDelta edges."""
    chat_prefix = [] if parent is None else parent.chat_completions
    prompt_prefix = [] if parent is None else [*parent.prompt_ids, *parent.response_ids]
    if not (
        parent is not None
        and (parent.metadata or {}).get("lineage_id") == (step.metadata or {}).get("lineage_id")
        and _messages_start_with(step.chat_completions, chat_prefix)
        and step.prompt_ids[: len(prompt_prefix)] == prompt_prefix
    ):
        parent, chat_prefix, prompt_prefix = None, [], []
    values = {name: getattr(step, name) for name in StepDelta.model_fields if name in Step.model_fields}
    return StepDelta(
        **values,
        parent_step_id=None if parent is None else parent.id,
        prompt_ids_suffix=step.prompt_ids[len(prompt_prefix) :],
        chat_completions_suffix=step.chat_completions[len(chat_prefix) :],
        finish_reason=getattr(step.model_output, "finish_reason", None),
    )


def _geometries():
    cumulative = _chain(1)
    interleaved = [item for pair in zip(_chain(100, "a"), _chain(200, "b"), strict=True) for item in pair]

    root_prefix = _chain(300)[:2]
    root_prefix[1] = _step(
        "main-reset",
        [*root_prefix[0].prompt_ids, *root_prefix[0].response_ids, 399],
        [499],
        [{"role": "system", "content": "new message path"}],
    )

    branch = _chain(500)[:2]
    branch.append(
        _step(
            "main-branch",
            [*branch[0].prompt_ids, *branch[0].response_ids, 599],
            [699],
            [*branch[0].chat_completions, {"role": "user", "content": "branch"}, {"role": "assistant", "content": "answer"}],
        )
    )
    return [
        _forms(cumulative),
        _forms(interleaved),
        _forms(root_prefix),
        _forms(branch, [None, 0, 0]),
    ]


def _datum_key(datum):
    return (
        list(datum.model_input.to_ints()),
        getattr(datum.model_input, "routing_matrices", None),
        {name: list(tensor.data) for name, tensor in datum.loss_fn_inputs.items()},
    )


def test_tinker_flat_delta_exact_parity():
    from rllm.trainer.tinker.transform import trajectory_to_datums

    for flat, delta in _geometries():
        for i, (step, compact_step) in enumerate(zip(flat.steps, delta.steps, strict=True)):
            step.routing_matrices = compact_step.routing_matrices = [f"route-{i}"]
        expected = [_datum_key(datum) for datum in trajectory_to_datums(flat, router_replay=True)]
        assert [_datum_key(datum) for datum in trajectory_to_datums(delta, router_replay=True)] == expected


def test_tinker_group_entry_flat_delta_exact_parity():
    from rllm.trainer.algorithms import AlgorithmConfig
    from rllm.trainer.tinker.transform import transform_trajectory_groups_to_datums

    flat, delta = _forms(_chain(1))
    for i, (step, compact_step) in enumerate(zip(flat.steps, delta.steps, strict=True)):
        step.routing_matrices = compact_step.routing_matrices = [f"route-{i}"]
    config = AlgorithmConfig(router_replay="R3")
    expected, expected_metrics = transform_trajectory_groups_to_datums([TrajectoryGroup(trajectories=[flat], group_id="task:role")], config)
    actual, actual_metrics = transform_trajectory_groups_to_datums([TrajectoryGroup(trajectories=[delta], group_id="task:role")], config)
    assert [_datum_key(datum) for datum in actual] == [_datum_key(datum) for datum in expected]
    assert actual_metrics == expected_metrics


def test_tinker_group_entry_oov_drop_parity():
    from rllm.trainer.algorithms import AlgorithmConfig
    from rllm.trainer.tinker.transform import transform_trajectory_groups_to_datums

    steps = _chain(1)
    steps[-1].prompt_ids.append(1_000)
    steps[-1].model_output.prompt_ids.append(1_000)
    flat, delta = _forms(steps)
    expected = transform_trajectory_groups_to_datums([TrajectoryGroup(trajectories=[flat], group_id="task:role")], AlgorithmConfig(), vocab_size=100)
    actual = transform_trajectory_groups_to_datums([TrajectoryGroup(trajectories=[delta], group_id="task:role")], AlgorithmConfig(), vocab_size=100)
    assert actual == expected
    assert actual[1]["batch/dropped_oov_sequences"] == 1


def test_tinker_direct_edges_do_not_compare_full_prefix(monkeypatch):
    import rllm.trainer.tinker.transform as transform

    flat, delta = _forms(_chain(1))
    expected = [_datum_key(datum) for datum in transform.trajectory_to_datums(flat)]
    resolve = transform._resolve_step_delta_prompt
    monkeypatch.setattr(
        transform,
        "_resolve_step_delta_prompt",
        lambda step, index: resolve(step, index) if step.parent_step_id is None else (_ for _ in ()).throw(AssertionError("full resolve")),
    )
    monkeypatch.setattr(transform, "_is_prefix", lambda *_: (_ for _ in ()).throw(AssertionError("full compare")))
    assert [_datum_key(datum) for datum in transform.trajectory_to_datums(delta)] == expected


def test_buffer_segment_count_flat_delta_parity():
    from rllm.trainer.buffer import TrajectoryGroupBuffer

    for flat, delta in _geometries():
        assert TrajectoryGroupBuffer._segment_count(delta) == TrajectoryGroupBuffer._segment_count(flat)


@pytest.fixture
def verl_transform(monkeypatch):
    """Load the converter against deterministic verl stubs; the helpers below cannot normalize a real DataProto."""
    verl = ModuleType("verl")
    verl.__path__ = []
    protocol = ModuleType("verl.protocol")

    class DataProto:
        @classmethod
        def from_dict(cls, tensors, non_tensors, meta_info):
            value = cls()
            value.batch = tensors
            value.non_tensor_batch = non_tensors
            value.meta_info = meta_info
            return value

    protocol.DataProto = DataProto
    utils = ModuleType("verl.utils")
    utils.__path__ = []
    torch_functional = ModuleType("verl.utils.torch_functional")

    def pad_sequence_to_length(value, max_length, pad_value, left_pad=False):
        if value.shape[1] >= max_length:
            return value
        padding = torch.full((value.shape[0], max_length - value.shape[1]), pad_value, dtype=value.dtype)
        return torch.cat((padding, value), dim=1) if left_pad else torch.cat((value, padding), dim=1)

    torch_functional.pad_sequence_to_length = pad_sequence_to_length
    for name, stub in {
        "verl": verl,
        "verl.protocol": protocol,
        "verl.utils": utils,
        "verl.utils.torch_functional": torch_functional,
    }.items():
        monkeypatch.setitem(sys.modules, name, stub)
    import rllm.engine.rollout as rollout

    monkeypatch.setattr(rollout, "VerlEngine", object, raising=False)
    sys.modules.pop("rllm.trainer.verl.transform", None)
    module = importlib.import_module("rllm.trainer.verl.transform")
    try:
        yield module
    finally:
        sys.modules.pop("rllm.trainer.verl.transform", None)
        package = sys.modules.get("rllm.trainer.verl")
        if getattr(package, "transform", None) is module:
            delattr(package, "transform")


def _routing(length, seed):
    values = np.arange(seed, seed + length * 2, dtype=np.int16).reshape(length, 2, 1)
    return [json.dumps({"shape": [2, 1], "dtype": "int16"}), base64.b64encode(values.tobytes()).decode()]


def _normal(value):
    if isinstance(value, torch.Tensor):
        return str(value.dtype), value.tolist()
    if isinstance(value, np.ndarray):
        return str(value.dtype), _normal(value.tolist())
    if isinstance(value, list):
        return [_normal(item) for item in value]
    if isinstance(value, dict):
        return {key: _normal(item) for key, item in value.items()}
    return value


def _verl_key(transform, trajectory):
    accumulated = transform.AccumulatedData()
    episode = Episode(id="task:0", trajectories=[trajectory], is_correct=True, metrics={"metric": 1})
    accumulated.repeat_counts.append(transform._process_episode(episode, "task", accumulated))
    return {name: _normal(getattr(accumulated, name)) for name in accumulated.__dataclass_fields__}


def _verl_batch_key(transform, trajectory, processor=None):
    engine = type("Engine", (), {"tokenizer": type("Tokenizer", (), {"pad_token_id": 0})(), "processor": processor})()
    episode = Episode(id="task:0", trajectories=[trajectory], is_correct=True, metrics={"metric": 1})
    batch = transform.transform_episodes_to_dataproto([episode], engine, max_prompt_length=32, max_response_length=32)
    return _normal(batch.batch), _normal(batch.non_tensor_batch), _normal(batch.meta_info)


def _verl_group_batch_key(transform, trajectory):
    engine = type("Engine", (), {"tokenizer": type("Tokenizer", (), {"pad_token_id": 0})(), "processor": None})()
    group = TrajectoryGroup(trajectories=[trajectory], group_id="task:role")
    batch = transform.transform_trajectory_groups_to_dataproto([group], engine, max_prompt_length=32, max_response_length=32)
    return _normal(batch.batch), _normal(batch.non_tensor_batch), _normal(batch.meta_info)


def test_verl_flat_delta_exact_parity_without_optional_dependency(verl_transform, monkeypatch):
    monkeypatch.setattr(verl_transform.uuid, "uuid4", lambda: "batch-id")
    for flat, delta in _geometries():
        for i, (step, compact_step) in enumerate(zip(flat.steps, delta.steps, strict=True)):
            routing = _routing(len(step.prompt_ids) + len(step.response_ids), i * 100)
            step.routing_matrices = compact_step.routing_matrices = routing
        assert _verl_key(verl_transform, delta) == _verl_key(verl_transform, flat)
        assert _verl_batch_key(verl_transform, delta) == _verl_batch_key(verl_transform, flat)
        assert _verl_group_batch_key(verl_transform, delta) == _verl_group_batch_key(verl_transform, flat)
        assert _verl_batch_key(verl_transform, delta, object()) == _verl_batch_key(verl_transform, flat, object())


def test_verl_skips_flat_step_without_model_output(verl_transform):
    step = Step(id="missing", prompt_ids=[1], response_ids=[2], logprobs=[-0.1])
    assert _verl_key(verl_transform, Trajectory(uid="trajectory", steps=[step], reward=1.0))["repeat_counts"] == [0]


def test_graph_enrichment_stays_compact_and_matches_flat():
    from rllm_model_gateway.models import TraceGraph, TraceRecord

    graph = TraceGraph(format="compact", version=1, deltas=[])
    records = [
        TraceRecord(
            trace_id="t0",
            session_id="session",
            lineage_id="main",
            messages=[{"role": "user", "content": "u0"}],
            prompt_token_ids=[1],
            response_message={"role": "assistant", "content": "a0"},
            completion_token_ids=[2],
            logprobs=[-0.1],
        ),
        TraceRecord(
            trace_id="t1",
            session_id="session",
            lineage_id="main",
            messages=[
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
                {"role": "user", "content": "u1"},
            ],
            prompt_token_ids=[1, 2, 3],
            response_message={"role": "assistant", "content": "a1"},
            completion_token_ids=[4],
            logprobs=[-0.2],
        ),
    ]
    for record in records:
        graph.add(record)
    shell = Episode(id="task:0", trajectories=[Trajectory(steps=[Step(reward=0.1), Step(reward=0.2)])])
    flat, compact = _enrichment_forms(graph, shell)
    assert isinstance(compact.trajectories[0], TrajectoryDelta)
    assert compact.trajectories[0].resolve().model_dump() == flat.trajectories[0].model_dump()


def _enrichment_forms(graph, shell):
    from rllm.engine.agentflow_engine import enrich_episode_with_traces

    flat = enrich_episode_with_traces(shell, graph.flatten(), "task:0", {}, strict=True)
    compact = enrich_episode_with_traces(shell, graph, "task:0", {}, strict=True)
    assert compact.metrics == flat.metrics
    return flat, compact


def test_graph_enrichment_reroots_after_filtered_middle_parent():
    from rllm_model_gateway.models import TraceGraph, TraceRecord

    graph = TraceGraph(format="compact", version=1, deltas=[])
    records = [
        TraceRecord(
            trace_id="t0", session_id="s", messages=[{"role": "user", "content": "u0"}], prompt_token_ids=[1], response_message={"role": "assistant", "content": "a0"}, completion_token_ids=[2]
        ),
        TraceRecord(
            trace_id="empty",
            session_id="s",
            messages=[{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}, {"role": "user", "content": "failed"}],
            prompt_token_ids=[1, 2, 3],
            response_message={},
            completion_token_ids=[],
        ),
        TraceRecord(
            trace_id="t2",
            session_id="s",
            messages=[{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}, {"role": "user", "content": "failed"}, {}, {"role": "user", "content": "u2"}],
            prompt_token_ids=[1, 2, 3, 5],
            response_message={"role": "assistant", "content": "a2"},
            completion_token_ids=[6],
        ),
    ]
    for record in records:
        graph.add(record)
    flat, compact = _enrichment_forms(graph, Episode(id="task:0", trajectories=[Trajectory(steps=[Step(), Step()])]))
    assert compact.trajectories[0].resolve().model_dump() == flat.trajectories[0].model_dump()
    assert compact.trajectories[0].steps[1].parent_step_id is None


def test_graph_enrichment_reroots_at_trajectory_boundary():
    from rllm_model_gateway.models import TraceGraph, TraceRecord

    graph = TraceGraph(format="compact", version=1, deltas=[])
    graph.add(
        TraceRecord(
            trace_id="t0", session_id="s", messages=[{"role": "user", "content": "u0"}], prompt_token_ids=[1], response_message={"role": "assistant", "content": "a0"}, completion_token_ids=[2]
        )
    )
    graph.add(
        TraceRecord(
            trace_id="t1",
            session_id="s",
            messages=[{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}, {"role": "user", "content": "u1"}],
            prompt_token_ids=[1, 2, 3],
            response_message={"role": "assistant", "content": "a1"},
            completion_token_ids=[4],
        )
    )
    flat, compact = _enrichment_forms(graph, Episode(id="task:0", trajectories=[Trajectory(steps=[Step()]), Trajectory(steps=[Step()])]))
    assert [trajectory.resolve().model_dump() for trajectory in compact.trajectories] == [trajectory.model_dump() for trajectory in flat.trajectories]
    assert compact.trajectories[1].steps[0].parent_step_id is None


def test_graph_enrichment_ancestry_ignores_arbitrary_metadata_lineage():
    from rllm_model_gateway.models import TraceGraph, TraceRecord

    graph = TraceGraph(format="compact", version=1, deltas=[])
    graph.add(
        TraceRecord(
            trace_id="t0",
            session_id="s",
            lineage_id=None,
            messages=[{"role": "user", "content": "u0"}],
            prompt_token_ids=[1],
            response_message={"role": "assistant", "content": "a0"},
            completion_token_ids=[2],
            metadata={"lineage_id": "metadata-a"},
        )
    )
    graph.add(
        TraceRecord(
            trace_id="t1",
            session_id="s",
            lineage_id=None,
            messages=[{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}, {"role": "user", "content": "u1"}],
            prompt_token_ids=[1, 2, 3],
            response_message={"role": "assistant", "content": "a1"},
            completion_token_ids=[4],
            metadata={"lineage_id": "metadata-b"},
        )
    )
    assert graph.deltas[1].parent_trace_id == "t0"
    flat, compact = _enrichment_forms(graph, Episode(id="task:0", trajectories=[Trajectory(steps=[Step(), Step()])]))
    assert compact.trajectories[0].resolve().model_dump() == flat.trajectories[0].model_dump()


def test_finish_episode_preserves_flat_evaluator_contract():
    from types import SimpleNamespace

    from rllm_model_gateway.models import TraceGraph, TraceRecord

    from rllm.engine.agentflow_engine import AgentFlowEngine, TaskContext
    from rllm.eval.types import EvalOutput
    from rllm.types import Task

    graph = TraceGraph(format="compact", version=1, deltas=[])
    graph.add(
        TraceRecord(
            trace_id="t0",
            session_id="s",
            messages=[{"role": "user", "content": "question"}],
            prompt_token_ids=[1],
            response_message={"role": "assistant", "content": "answer"},
            completion_token_ids=[2],
            logprobs=[-0.1],
        )
    )

    class Evaluator:
        def evaluate(self, _task, episode):
            trajectory = episode.trajectories[0]
            assert isinstance(trajectory, Trajectory)
            assert trajectory.steps[0].prompt_ids == [1]
            assert trajectory.steps[0].chat_completions[-1]["content"] == "answer"
            assert trajectory.steps[0].model_output.content == "answer"
            trajectory.reward = 0.75
            trajectory.signals = {"custom": 0.25}
            return EvalOutput(reward=0.5, is_correct=True)

    engine = object.__new__(AgentFlowEngine)
    engine.executor = None
    engine.agent_flow = SimpleNamespace(makes_llm_calls=False)
    result = asyncio.run(
        engine._finish_episode(
            Episode(id="task:0", trajectories=[Trajectory(steps=[Step()])]),
            graph,
            "task:0",
            Task(id="task", instruction="question"),
            TaskContext(evaluator=Evaluator()),
        )
    )
    assert isinstance(result.trajectories[0], TrajectoryDelta)
    assert result.trajectories[0].reward == 0.75
    assert result.trajectories[0].signals == {"custom": 0.25}


def test_manager_keeps_legacy_fetch_flat_and_exposes_async_graph_fetch():
    from rllm_model_gateway.models import TraceGraph

    from rllm.gateway.manager import GatewayManager

    graph = TraceGraph(format="compact", version=1, deltas=[])

    class SyncClient:
        def __init__(self):
            self.calls = []

        def flush(self):
            pass

        def get_session_traces(self, _session_id, **kwargs):
            self.calls.append(kwargs)
            return graph if kwargs.get("flatten") is False else []

    class AsyncClient(SyncClient):
        async def flush(self, **_kwargs):
            pass

        async def get_session_traces(self, session_id, **kwargs):
            return super().get_session_traces(session_id, **kwargs)

    manager = object.__new__(GatewayManager)
    manager.store = "compact"
    manager._client = SyncClient()
    assert manager.get_traces("s") == []
    manager._async_client = AsyncClient()

    async def fetch():
        assert await manager.aget_traces("s") == []
        assert await manager.aget_trace_graph("s") is graph

    asyncio.run(fetch())
    assert manager.client.calls == [{"format": "compact"}]
    assert manager.async_client.calls == [{"format": "compact"}, {"format": "compact", "flatten": False}]


def test_episode_delta_json_and_legacy_dict_roundtrip():
    flat, delta = _forms(_chain(1))
    delta.task = {"image": object(), "kept": 1}
    delta.steps[0].action = Action(action="go")
    delta.metadata = {"source": "compact"}
    episode = Episode(id="task:0", trajectories=[delta])
    wire_episode = episode.model_copy(update={"trajectories": [delta.model_copy(update={"task": None})]})
    assert isinstance(Episode.model_validate_json(wire_episode.model_dump_json()).trajectories[0], TrajectoryDelta)
    assert isinstance(Episode.model_validate_json(Episode(id="task:0", trajectories=[flat]).model_dump_json()).trajectories[0], Trajectory)
    assert isinstance(Episode.model_validate_json(Episode(id="task:0", trajectories=[Trajectory()]).model_dump_json()).trajectories[0], Trajectory)
    delta.steps[0].input = delta.steps[0].output = object()
    payload = episode.to_dict()
    assert payload["trajectories"][0]["steps"][0]["action"] == "go"
    assert "input" not in payload["trajectories"][0]["steps"][0]
    assert "output" not in payload["trajectories"][0]["steps"][0]
    payload.pop("termination_reason")
    restored = Episode.from_dict(payload).trajectories[0]
    assert isinstance(restored, TrajectoryDelta)
    assert restored.task == {"kept": 1}
    assert restored.steps[0].action == "go"
    assert restored.metadata == delta.metadata
    flat.steps[0].action = "go"
    assert restored.resolve().model_copy(update={"task": None, "metadata": None}).model_dump() == flat.model_dump()


@pytest.mark.parametrize("container", [Episode, TrajectoryGroup])
def test_malformed_delta_cannot_fall_through_to_flat_union(container):
    _, delta = _forms(_chain(1))
    trajectory = delta.model_dump(mode="json")
    trajectory["steps"][0]["prompt_ids_suffix"] = "bad"
    with pytest.raises(ValueError):
        container.model_validate({"trajectories": [trajectory]})


def test_parent_ids_do_not_depend_on_arbitrary_metadata_lineage():
    root = StepDelta(id="root", parent_step_id=None, prompt_ids_suffix=[1], chat_completions_suffix=[], response_ids=[2], metadata={"lineage_id": "metadata-a"})
    child = StepDelta(id="child", parent_step_id="root", prompt_ids_suffix=[3], chat_completions_suffix=[], response_ids=[4], metadata={"lineage_id": "metadata-b"})
    assert _index_step_deltas([root, child])[1]["child"] == 3
