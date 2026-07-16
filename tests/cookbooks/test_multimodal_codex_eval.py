"""Unit tests for cookbooks.multimodal_codex.multimodal_codex_eval."""

from __future__ import annotations

from rllm.types import Episode, Step, Task, Trajectory

from cookbooks.multimodal_codex.multimodal_codex_eval import (
    _extract_final_content,
    _matches,
    _normalize,
    multimodal_codex_evaluator,
)


def _make_episode_with_output(output: str) -> Episode:
    return Episode(
        task={},
        trajectories=[
            Trajectory(name="codex", steps=[Step(output=output)]),
        ],
    )


def _make_task(gt: str, task_type: str = "chart_reading") -> Task:
    return Task(
        id="t1",
        instruction="what is the value?",
        metadata={"ground_truth": gt, "task_type": task_type},
    )


def test_json_agent_message_exact_match():
    stdout = "\n".join(
        [
            '{"type":"thinking","content":"I need to look at the chart..."}',
            '{"type":"agent_message","message":{"role":"assistant","content":"42"}}',
            '{"type":"turn_complete"}',
        ]
    )
    result = multimodal_codex_evaluator(_make_task("42"), _make_episode_with_output(stdout))
    assert result.reward == 1.0
    assert result.is_correct is True
    assert result.metadata["model_answer"] == "42"


def test_json_wrong_answer():
    stdout = '{"type":"agent_message","message":{"role":"assistant","content":"43"}}\n'
    result = multimodal_codex_evaluator(_make_task("42"), _make_episode_with_output(stdout))
    assert result.reward == 0.0
    assert result.is_correct is False


def test_fallback_substring_match():
    # Not valid JSON — evaluator falls back to substring match at word boundary.
    stdout = "The answer is 42 based on the chart."
    result = multimodal_codex_evaluator(_make_task("42"), _make_episode_with_output(stdout))
    assert result.reward == 1.0
    assert result.is_correct is True


def test_no_answer_returns_zero():
    stdout = "Some unrelated debugging output about tools and paths."
    result = multimodal_codex_evaluator(_make_task("42"), _make_episode_with_output(stdout))
    assert result.reward == 0.0
    assert result.is_correct is False


def test_no_ground_truth_returns_zero():
    task = Task(id="t1", instruction="q", metadata={"task_type": "chart_reading"})
    ep = _make_episode_with_output('{"type":"agent_message","content":"42"}')
    result = multimodal_codex_evaluator(task, ep)
    assert result.reward == 0.0
    assert "no ground_truth" in result.metadata.get("error", "")


def test_flat_content_schema():
    # Alternative Codex schema: content at top level, not nested in message.
    stdout = '{"type":"agent_message","content":"7"}\n'
    result = multimodal_codex_evaluator(_make_task("7"), _make_episode_with_output(stdout))
    assert result.reward == 1.0


def test_last_agent_message_wins():
    # If multiple agent_messages exist within a single step, use the LAST one.
    # Content is bare "42" — primary path uses strict equality (rejects
    # negations like "not 3, but 5") so wrapped answers like "Actually, 42."
    # deliberately fail; that's the anti-B3 behavior tested separately.
    stdout = "\n".join(
        [
            '{"type":"agent_message","content":"Let me think..."}',
            '{"type":"agent_message","content":"42"}',
        ]
    )
    result = multimodal_codex_evaluator(_make_task("42"), _make_episode_with_output(stdout))
    assert result.reward == 1.0
    assert result.metadata["model_answer"] == "42"


def test_word_boundary_no_false_positive():
    # gt="5" should NOT match "15" — word boundary guardrail.
    stdout = "I count 15 shapes."
    result = multimodal_codex_evaluator(_make_task("5"), _make_episode_with_output(stdout))
    assert result.reward == 0.0


def test_extract_final_content_returns_none_on_no_events():
    assert _extract_final_content("") is None
    assert _extract_final_content("not json at all") is None
    assert _extract_final_content('{"type":"thinking","content":"..."}') is None


def test_normalize_strips_punctuation():
    assert _normalize("  Hello.  ") == "hello"
    assert _normalize("42") == "42"


def test_matches_case_insensitive():
    assert _matches("HELLO", "hello") is True
    assert _matches("hello", "world") is False


def test_late_step_agent_message_wins():
    """Cross-step last-wins: the FINAL step's agent_message beats an earlier
    intermediate one. Codex may 'think out loud' in step 0 before delivering
    the real answer in a later step."""
    ep = Episode(
        task={},
        trajectories=[
            Trajectory(
                name="codex",
                steps=[
                    Step(output='{"type":"agent_message","content":"Let me examine the chart..."}'),
                    Step(output='{"type":"agent_message","content":"42"}'),
                ],
            ),
        ],
    )
    result = multimodal_codex_evaluator(_make_task("42"), ep)
    assert result.reward == 1.0
    assert result.metadata["model_answer"] == "42"


def test_negation_not_false_positive():
    """Primary path (has agent_message) uses STRICT equality — a negation
    like 'not 3, but 5' must NOT be scored as gt='3' correct."""
    stdout = '{"type":"agent_message","content":"not 3, but 5"}'
    result = multimodal_codex_evaluator(_make_task("3"), _make_episode_with_output(stdout))
    assert result.reward == 0.0
    assert result.is_correct is False
