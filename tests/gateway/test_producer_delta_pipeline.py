"""Producer flip: the compact fetch keeps prompt-id deltas in step form and
the whole pipeline consumes them without materializing O(n^2) token lists.

The expanded path is retained ONLY as the parity instrument: the same store
contents fetched both ways must produce identical steps (up to form) and —
via the packed builder — identical training datums.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rllm-model-gateway" / "src"))

from rllm_model_gateway.client import _expand_compact_traces  # noqa: E402
from rllm_model_gateway.models import TraceRecord  # noqa: E402
from rllm_model_gateway.store.memory_store import MemoryTraceStore  # noqa: E402

from rllm.engine.trace_converter import compute_step_metrics, trace_record_to_step  # noqa: E402
from rllm.types import Trajectory  # noqa: E402


def _store_session(lineages=1, turns=6):
    """A compact store holding cumulative conversations with growing ids."""
    store = MemoryTraceStore(compact=True)

    async def fill():
        for lin in range(lineages):
            ids: list[int] = [7000 + lin, 7001 + lin]
            messages = [{"role": "user", "content": f"q-{lin}"}]
            for turn in range(turns):
                ids = ids + [100 * lin + turn, 100 * lin + turn + 1]
                await store.store_trace(
                    f"t-{lin}-{turn}",
                    "s",
                    {
                        "trace_id": f"t-{lin}-{turn}",
                        "session_id": "s",
                        "lineage_id": f"lin{lin}",
                        "messages": list(messages),
                        "response_message": {"role": "assistant", "content": f"a-{lin}-{turn}"},
                        "prompt_token_ids": list(ids),
                        "completion_token_ids": [900 + turn, 901 + turn],
                        "logprobs": [-0.1, -0.2],
                    },
                )
                messages = messages + [
                    {"role": "assistant", "content": f"a-{lin}-{turn}"},
                    {"role": "user", "content": f"q-{lin}-{turn}"},
                ]
                ids = ids + [900 + turn, 901 + turn]  # + response, cumulative
        return json.loads(json.dumps(await store.get_session_traces_compact("s"), default=str))

    return asyncio.run(fill())


def _resolve_marker_ids(traces):
    """Reference resolution of kept markers back to full lists, per lineage."""
    cur: dict = {}
    out = []
    for t in traces:
        ids = t["prompt_token_ids"]
        lin = t.get("lineage_id")
        if isinstance(ids, dict):
            lcp, suffix = ids["__prompt_ids_delta__"]
            cur[lin] = cur.get(lin, [])[:lcp] + list(suffix)
        else:
            cur[lin] = list(ids)
        out.append(cur[lin])
    return out


def test_kept_deltas_resolve_to_the_expanded_lists():
    payload = _store_session(lineages=2, turns=6)
    expanded = _expand_compact_traces(payload)
    kept = _expand_compact_traces(payload, expand_prompt_ids=False)
    assert [t["trace_id"] for t in kept] == [t["trace_id"] for t in expanded]
    # every non-root trace kept its ids as the step-form marker
    marked = [t for t in kept if isinstance(t["prompt_token_ids"], dict)]
    assert len(marked) == len(kept) - 2  # one full-list root per lineage
    assert _resolve_marker_ids(kept) == [t["prompt_token_ids"] for t in expanded]


def test_kept_delta_memory_is_linear_not_quadratic():
    payload = _store_session(lineages=1, turns=40)
    expanded = _expand_compact_traces(payload)
    kept = _expand_compact_traces(payload, expand_prompt_ids=False)
    n_expanded = sum(len(t["prompt_token_ids"]) for t in expanded)
    n_kept = sum(len(t["prompt_token_ids"]["__prompt_ids_delta__"][1]) if isinstance(t["prompt_token_ids"], dict) else len(t["prompt_token_ids"]) for t in kept)
    assert n_expanded > 5 * n_kept  # quadratic vs linear at 40 turns


def test_chain_not_lineage_predecessor_rebases_to_full_list():
    """Defensive path: a delta whose ancestor is not the lineage's previous
    trace must be rebased to a full list, never kept as a broken marker."""
    payload = {
        "format": "compact",
        "nodes": {"n1": {"p": None, "m": {"role": "user", "content": "x"}}},
        "traces": [
            {"trace_id": "a", "_tid": "a", "lineage_id": None, "messages_ref": ["n1", 1], "prompt_token_ids": [1, 2, 3], "response_message": {}},
            {"trace_id": "b", "_tid": "b", "lineage_id": None, "messages_ref": ["n1", 1], "prompt_ids_delta": ["a", 3, [4]], "response_message": {}},
            # c's chain ancestor is a, but its lineage predecessor is b
            {"trace_id": "c", "_tid": "c", "lineage_id": None, "messages_ref": ["n1", 1], "prompt_ids_delta": ["a", 2, [9]], "response_message": {}},
        ],
    }
    kept = _expand_compact_traces(payload, expand_prompt_ids=False)
    assert kept[1]["prompt_token_ids"] == {"__prompt_ids_delta__": [3, [4]]}
    assert kept[2]["prompt_token_ids"] == [1, 2, 9]  # rebased, exact


def test_trace_record_to_step_keeps_marker_and_exact_length():
    payload = _store_session(lineages=1, turns=4)
    kept = _expand_compact_traces(payload, expand_prompt_ids=False)
    expanded = _expand_compact_traces(payload)
    for k, e in zip(kept, expanded, strict=True):
        step = trace_record_to_step(TraceRecord.model_construct(**k))
        assert step.prompt_len == len(e["prompt_token_ids"])  # exact, no expansion
        if isinstance(k["prompt_token_ids"], dict):
            assert step.prompt_delta is not None
            assert step.model_output.prompt_ids is None  # nothing materialized
            assert step.model_output.prompt_length == len(e["prompt_token_ids"])
        else:
            assert step.prompt_delta is None
            assert list(step.prompt_ids) == list(e["prompt_token_ids"])


def test_step_metrics_identical_both_forms():
    payload = _store_session(lineages=1, turns=5)
    mk = lambda traces: Trajectory(steps=[trace_record_to_step(TraceRecord.model_construct(**t)) for t in traces])  # noqa: E731
    m_kept = compute_step_metrics([mk(_expand_compact_traces(payload, expand_prompt_ids=False))])
    m_full = compute_step_metrics([mk(_expand_compact_traces(payload))])
    assert m_kept == m_full


def test_end_to_end_datums_identical_both_forms():
    """THE gate: store → keep-mode fetch → steps → packed datums must equal
    the expanded-path datums exactly. The expansion exists only as this
    reference."""
    tinker = pytest.importorskip("tinker")  # noqa: F841
    from rllm.trainer.tinker.transform import trajectory_to_datums

    payload = _store_session(lineages=2, turns=6)

    def datums(expand):
        traces = _expand_compact_traces(payload, expand_prompt_ids=expand)
        steps = [trace_record_to_step(TraceRecord.model_construct(**t)) for t in traces]
        for s in steps:
            s.advantage = 0.5
        return trajectory_to_datums(Trajectory(steps=steps))

    ref, new = datums(True), datums(False)
    assert len(ref) == len(new) == 2  # one packed datum per lineage
    for a, b in zip(ref, new, strict=True):
        assert list(a.model_input.to_ints()) == list(b.model_input.to_ints())
        for key in a.loss_fn_inputs:
            assert list(a.loss_fn_inputs[key].data) == list(b.loss_fn_inputs[key].data), key
