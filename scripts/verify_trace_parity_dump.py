#!/usr/bin/env python3
"""Verify that dumped compact gateway graphs exactly reproduce raw records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from rllm_model_gateway.models import TraceGraph, TraceRecord


def _session_id(generation_dir: Path) -> str:
    manifest = generation_dir.parent / "session.json"
    if not manifest.exists():
        return generation_dir.parent.name
    return str(json.loads(manifest.read_text(encoding="utf-8"))["session_id"])


def _read_raw_records(path: Path) -> list[TraceRecord]:
    if not path.exists():
        return []
    records: list[TraceRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(TraceRecord.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: invalid TraceRecord: {exc}") from exc
    return records


def _final_raw_records(records: list[TraceRecord]) -> list[TraceRecord]:
    # Compact-store retries replace the same leaf in place. Assignment to an
    # existing OrderedDict key retains that trace's original graph position.
    final: OrderedDict[str, TraceRecord] = OrderedDict()
    for record in records:
        final[record.trace_id] = record
    return list(final.values())


@dataclass
class _ExpectedTrace:
    record: TraceRecord
    parent_trace_id: str | None


def _message_key(message: dict) -> str:
    """Compare chat blocks byte-semantically, including dict key order."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _expected_canonical_message(message: dict) -> dict:
    """Independently model the raw-message-to-graph normalization contract.

    Keep this separate from the production converter: sharing that helper would
    make the parity check unable to detect a bug in the conversion itself.
    """
    normalized = json.loads(_message_key(message))
    provider_fields = normalized.get("provider_specific_fields")
    if provider_fields is None:
        normalized.pop("provider_specific_fields", None)
        provider_fields = None

    reasoning_values = [value for value in (normalized.get("reasoning"), normalized.get("reasoning_content")) if value is not None]
    if reasoning_values and all(value == reasoning_values[0] for value in reasoning_values[1:]):
        normalized.pop("reasoning", None)
        normalized.pop("reasoning_content", None)
        normalized["reasoning_content"] = reasoning_values[0]
        if isinstance(provider_fields, dict) and provider_fields.get("reasoning") == reasoning_values[0]:
            provider_fields.pop("reasoning", None)
    elif not reasoning_values:
        if normalized.get("reasoning") is None:
            normalized.pop("reasoning", None)
        if normalized.get("reasoning_content") is None:
            normalized.pop("reasoning_content", None)
        if isinstance(provider_fields, dict) and provider_fields.get("reasoning") is None:
            provider_fields.pop("reasoning", None)

    if normalized.get("refusal") is None:
        normalized.pop("refusal", None)
    if isinstance(provider_fields, dict):
        if provider_fields.get("refusal") is None:
            provider_fields.pop("refusal", None)
        if not provider_fields:
            normalized.pop("provider_specific_fields", None)
    return json.loads(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def _expected_canonical_record(record: TraceRecord) -> TraceRecord:
    return record.model_copy(
        update={
            "messages": [_expected_canonical_message(message) for message in record.messages],
            "response_message": _expected_canonical_message(record.response_message),
        }
    )


def _messages_start_with(values: list[dict], prefix: list[dict]) -> bool:
    return len(prefix) <= len(values) and all(_message_key(value) == _message_key(expected) for value, expected in zip(values[: len(prefix)], prefix, strict=True))


def _completed_messages(record: TraceRecord) -> list[dict]:
    return [*record.messages, record.response_message]


def _completed_token_ids(record: TraceRecord) -> list[int]:
    return [*record.prompt_token_ids, *record.completion_token_ids]


def _valid_parent_prefix(parent: TraceRecord, record: TraceRecord, *, require_same_model: bool) -> bool:
    return (
        parent.session_id == record.session_id
        and parent.lineage_id == record.lineage_id
        and (not require_same_model or parent.model == record.model)
        and _messages_start_with(record.messages, _completed_messages(parent))
        and record.prompt_token_ids[: len(_completed_token_ids(parent))] == _completed_token_ids(parent)
    )


def _find_expected_parent(record: TraceRecord, states: OrderedDict[str, _ExpectedTrace]) -> str | None:
    """Independently derive the deepest prior completed-message parent.

    Parent discovery is message-first. If the deepest matching completed state
    has a token-tape mismatch, compaction must safely make the record a root;
    it must not silently attach to a shallower state.
    """
    best: TraceRecord | None = None
    best_depth = -1
    for state in states.values():
        candidate = state.record
        completed_messages = _completed_messages(candidate)
        if (
            candidate.session_id == record.session_id
            and candidate.lineage_id == record.lineage_id
            and candidate.model == record.model
            and _messages_start_with(record.messages, completed_messages)
            and len(completed_messages) > best_depth
        ):
            # Strictly greater keeps the first inserted representative when
            # two records complete the same message path.
            best = candidate
            best_depth = len(completed_messages)
    if best is None:
        return None
    return best.trace_id if record.prompt_token_ids[: len(_completed_token_ids(best))] == _completed_token_ids(best) else None


def _expected_structure(records: list[TraceRecord]) -> list[_ExpectedTrace]:
    """Model add/leaf-replacement semantics from raw records, without a graph."""
    states: OrderedDict[str, _ExpectedTrace] = OrderedDict()
    for raw_record in records:
        record = _expected_canonical_record(raw_record)
        current = states.get(record.trace_id)
        if current is None:
            states[record.trace_id] = _ExpectedTrace(record=record, parent_trace_id=_find_expected_parent(record, states))
            continue

        if any(state.parent_trace_id == record.trace_id for state in states.values()):
            raise ValueError(f"raw input tries to replace non-leaf trace {record.trace_id!r}")
        if current.record.session_id != record.session_id or current.record.lineage_id != record.lineage_id:
            raise ValueError(f"raw input moves trace {record.trace_id!r} to another session or lineage")

        parent = states.get(current.parent_trace_id) if current.parent_trace_id is not None else None
        parent_trace_id = current.parent_trace_id if parent is not None and _valid_parent_prefix(parent.record, record, require_same_model=False) else None
        states[record.trace_id] = _ExpectedTrace(record=record, parent_trace_id=parent_trace_id)
    return list(states.values())


def _different_fields(expected: TraceRecord, actual: TraceRecord) -> list[str]:
    expected_data = expected.model_dump(mode="json")
    actual_data = actual.model_dump(mode="json")
    return sorted(key for key in expected_data.keys() | actual_data.keys() if expected_data.get(key) != actual_data.get(key))


def verify_generation(generation_dir: Path) -> tuple[bool, str, int, int, int]:
    session_id = _session_id(generation_dir)
    raw_path = generation_dir / "raw_trace_records.jsonl"
    graph_path = generation_dir / "trace_graph.json"
    raw_records = _read_raw_records(raw_path)
    try:
        expected_states = _expected_structure(raw_records)
    except Exception as exc:
        return False, f"{session_id} {generation_dir.name}: invalid raw store sequence: {exc}", len(_final_raw_records(raw_records)), raw_path.stat().st_size if raw_path.exists() else 0, 0
    expected = [state.record for state in expected_states]
    duplicates = len(raw_records) - len(expected)

    if not graph_path.exists():
        return False, f"{session_id} {generation_dir.name}: missing trace_graph.json", len(expected), raw_path.stat().st_size if raw_path.exists() else 0, 0

    try:
        graph = TraceGraph.model_validate_json(graph_path.read_text(encoding="utf-8"))
        actual = graph.flatten()
    except Exception as exc:
        return False, f"{session_id} {generation_dir.name}: invalid/unresolvable graph: {exc}", len(expected), raw_path.stat().st_size if raw_path.exists() else 0, graph_path.stat().st_size

    expected_ids = [record.trace_id for record in expected]
    actual_ids = [record.trace_id for record in actual]
    if expected_ids != actual_ids:
        return (
            False,
            f"{session_id} {generation_dir.name}: trace id/order mismatch raw={expected_ids} graph={actual_ids}",
            len(expected),
            raw_path.stat().st_size if raw_path.exists() else 0,
            graph_path.stat().st_size,
        )

    for expected_state, delta in zip(expected_states, graph.deltas, strict=True):
        parent = next((state.record for state in expected_states if state.record.trace_id == expected_state.parent_trace_id), None)
        expected_messages_suffix = expected_state.record.messages if parent is None else expected_state.record.messages[len(_completed_messages(parent)) :]
        expected_prompt_ids_suffix = expected_state.record.prompt_token_ids if parent is None else expected_state.record.prompt_token_ids[len(_completed_token_ids(parent)) :]
        if delta.parent_trace_id != expected_state.parent_trace_id:
            return (
                False,
                f"{session_id} {generation_dir.name}: trace {expected_state.record.trace_id!r} parent_trace_id raw-derived={expected_state.parent_trace_id!r} graph={delta.parent_trace_id!r}",
                len(expected),
                raw_path.stat().st_size if raw_path.exists() else 0,
                graph_path.stat().st_size,
            )
        if delta.messages_suffix != expected_messages_suffix:
            return (
                False,
                f"{session_id} {generation_dir.name}: trace {expected_state.record.trace_id!r} has incorrect messages_suffix",
                len(expected),
                raw_path.stat().st_size if raw_path.exists() else 0,
                graph_path.stat().st_size,
            )
        if delta.prompt_ids_suffix != expected_prompt_ids_suffix:
            return (
                False,
                f"{session_id} {generation_dir.name}: trace {expected_state.record.trace_id!r} has incorrect prompt_ids_suffix",
                len(expected),
                raw_path.stat().st_size if raw_path.exists() else 0,
                graph_path.stat().st_size,
            )

    for expected_record, actual_record in zip(expected, actual, strict=True):
        fields = _different_fields(expected_record, actual_record)
        if fields:
            return (
                False,
                f"{session_id} {generation_dir.name}: trace {expected_record.trace_id!r} differs in fields: {', '.join(fields)}",
                len(expected),
                raw_path.stat().st_size if raw_path.exists() else 0,
                graph_path.stat().st_size,
            )

    detail = f"{session_id} {generation_dir.name}: {len(expected)} records ({duplicates} replacement inputs)"
    return True, detail, len(expected), raw_path.stat().st_size if raw_path.exists() else 0, graph_path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="Directory configured as rllm.gateway.trace_parity_dump_dir")
    parser.add_argument("--verbose", action="store_true", help="Print each passing session generation")
    args = parser.parse_args()

    generation_dirs = sorted(path for path in args.dump_dir.glob("session-*/generation-*") if path.is_dir())
    if not generation_dirs:
        print(f"No trace parity generations found under {args.dump_dir}", file=sys.stderr)
        return 2

    failures: list[str] = []
    total_records = 0
    raw_bytes = 0
    graph_bytes = 0
    for generation_dir in generation_dirs:
        try:
            passed, detail, records, generation_raw_bytes, generation_graph_bytes = verify_generation(generation_dir)
        except Exception as exc:
            passed, detail, records, generation_raw_bytes, generation_graph_bytes = False, f"{generation_dir}: {exc}", 0, 0, 0
        total_records += records
        raw_bytes += generation_raw_bytes
        graph_bytes += generation_graph_bytes
        if not passed:
            failures.append(detail)
            print(f"FAIL {detail}")
        elif args.verbose:
            print(f"PASS {detail}")

    ratio = raw_bytes / graph_bytes if graph_bytes else float("inf")
    print(f"Checked {len(generation_dirs)} session generations and {total_records} final records: {len(failures)} failures; raw={raw_bytes} bytes graph={graph_bytes} bytes compression={ratio:.2f}x")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
