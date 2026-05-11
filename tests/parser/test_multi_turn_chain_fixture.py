"""Smoke test that pytest can import the fixture from a sibling test file
under tests/parser/. Will be removed once a real test consumes it.
"""

from _fixtures.multi_turn_chain import (
    CHAIN_ENDING_ON_TOOL,
    CHAIN_ENDING_ON_USER,
    get_chain,
    get_realistic_transitions,
    step_by_label,
)


def test_chain_shapes():
    assert len(CHAIN_ENDING_ON_TOOL) == 3
    assert len(CHAIN_ENDING_ON_USER) == 5
    assert CHAIN_ENDING_ON_TOOL[-1]["messages"][-1]["role"] == "tool"
    assert CHAIN_ENDING_ON_USER[-1]["messages"][-1]["role"] == "user"


def test_realistic_transitions_endpoints_are_engine_inputs():
    for variant in ("ending_on_tool", "ending_on_user"):
        chain = get_chain(variant)
        for src, dst in get_realistic_transitions(variant):
            for label in (src, dst):
                last = step_by_label(chain, label)["messages"][-1]
                assert last["role"] in {"user", "tool"}, f"{variant}:{label} ends on {last['role']}"


def test_chain_is_append_only_prefix_growing():
    for chain_name, chain in (("tool", CHAIN_ENDING_ON_TOOL), ("user", CHAIN_ENDING_ON_USER)):
        for i in range(len(chain) - 1):
            prev = chain[i]["messages"]
            curr = chain[i + 1]["messages"]
            assert len(curr) > len(prev), f"{chain_name} chain step {i} -> {i + 1}: not strictly growing"
            # The first len(prev) messages must be the same objects (append-only).
            assert curr[: len(prev)] == prev, f"{chain_name} chain step {i} -> {i + 1}: prefix divergence"
