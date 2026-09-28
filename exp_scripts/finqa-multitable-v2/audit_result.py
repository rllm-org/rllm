#!/usr/bin/env python3
"""Fail-closed completeness/provenance audit for one v2 evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROTOCOL_VERSION = "finqa-multitable-visible-complete-v2"
MAX_TURNS = 50
PROMPT_PATH = Path(__file__).parent / "prompts" / "multitable_v2_system_prompt.txt"
PROMPT_SHA256 = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("episodes_dir", type=Path)
    parser.add_argument("expected", type=int)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("total") != args.expected or len(result.get("items", [])) != args.expected:
        raise SystemExit(
            f"result incomplete: total={result.get('total')} items={len(result.get('items', []))} "
            f"expected={args.expected}"
        )

    paths = sorted((args.episodes_dir / "episodes").glob("episode_*.json"))
    if len(paths) != args.expected:
        raise SystemExit(f"episode count={len(paths)}, expected={args.expected}")

    finalize_reasons: dict[str, int] = {}
    empty_answers = fallback_answers = over_turns = 0
    prompt_mismatches = protocol_mismatches = 0
    tool_calls = tool_errors = malformed_tool_calls = llm_errors = 0
    turns: list[int] = []
    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        artifacts = episode.get("artifacts") or {}
        protocol_mismatches += artifacts.get("protocol_version") != PROTOCOL_VERSION
        prompt_mismatches += artifacts.get("prompt_sha256") != PROMPT_SHA256
        turn_count = int(artifacts.get("turns", -1))
        turns.append(turn_count)
        over_turns += turn_count > MAX_TURNS
        empty_answers += not bool(str(artifacts.get("answer", "")).strip())
        fallback_answers += bool(artifacts.get("final_fallback_used"))
        tool_calls += int(artifacts.get("tool_calls", 0))
        tool_errors += int(artifacts.get("tool_errors", 0))
        malformed_tool_calls += int(artifacts.get("malformed_tool_calls", 0))
        llm_errors += len(artifacts.get("llm_errors") or [])
        reason = str(artifacts.get("finalize_reason", "missing"))
        finalize_reasons[reason] = finalize_reasons.get(reason, 0) + 1

    hard_failures = {
        "protocol_mismatches": protocol_mismatches,
        "prompt_mismatches": prompt_mismatches,
        "over_turns": over_turns,
        "empty_answers": empty_answers,
    }
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "expected": args.expected,
        "result_total": result["total"],
        "result_errors": result.get("errors"),
        "episodes": len(paths),
        "mean_turns": sum(turns) / len(turns),
        "max_turns_observed": max(turns),
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "malformed_tool_calls": malformed_tool_calls,
        "llm_errors": llm_errors,
        "fallback_answers": fallback_answers,
        "finalize_reasons": finalize_reasons,
        **hard_failures,
    }
    audit_path = args.result.with_suffix(".protocol_audit.json")
    audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if any(hard_failures.values()):
        raise SystemExit(f"protocol audit failed: {hard_failures}")


if __name__ == "__main__":
    main()
