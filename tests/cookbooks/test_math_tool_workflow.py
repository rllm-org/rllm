"""End-to-end smoke test for ``cookbooks/math_tool_agent/workflow.py``.

Uses a stub ``RolloutEngine`` whose ``get_token_output_from_token_input``
returns pre-canned assistant turns (rendered via the same parser the
completer uses). The stub gives back:

  turn 0: assistant emits a tool call to evaluate ``12 * 7 + 3``
  turn 1: assistant emits the final boxed answer

The workflow should:
  1. Drive the loop through both turns,
  2. Execute the calculator inline on turn 0's tool call,
  3. Stop on turn 1 (no tool calls in the response),
  4. Extract the boxed answer "87",
  5. Compute reward 1.0 vs ``task["answer"] = "87"``,
  6. Commit a trajectory with 2 steps.
"""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

# Make the workflow module importable as a sibling — its parent dir isn't a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rollout"))


@dataclass
class _StubTokenOutput:
    token_ids: list[int]
    logprobs: list[float]


class _StubRolloutEngine:
    """Stub RolloutEngine that yields pre-canned completions per turn."""

    is_validation = False

    def __init__(self, tokenizer, chat_parser, canned_completions: list[list[int]]):
        self.tokenizer = tokenizer
        self.chat_parser = chat_parser
        self._canned = list(canned_completions)
        self.weight_version = 0

    @property
    def supports_token_in_token_out(self) -> bool:
        return True

    async def get_token_output_from_token_input(self, token_input, **kwargs):
        if not self._canned:
            raise RuntimeError("stub rollout: ran out of canned completions")
        completion_ids = self._canned.pop(0)
        return _StubTokenOutput(token_ids=completion_ids, logprobs=[0.0] * len(completion_ids))

    def assemble_model_output(self, token_input, token_output):
        from rllm.experimental.rollout.rollout_engine import ModelOutput

        text = self.tokenizer.decode(token_output.token_ids, skip_special_tokens=False)
        # Roundtrip the body through the chat parser's parse_completion so we get
        # back (content, reasoning, tool_calls).
        parsed = self.chat_parser.parse_completion(token_output.token_ids)
        return ModelOutput(
            text=text,
            content=parsed.get("content"),
            reasoning=parsed.get("reasoning"),
            tool_calls=parsed.get("tool_calls") or None,
            prompt_ids=token_input,
            completion_ids=token_output.token_ids,
            prompt_length=len(token_input),
            completion_length=len(token_output.token_ids),
            finish_reason="stop",
        )


def _load_parser_and_tokenizer():
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
    except ImportError:
        pytest.skip("transformers not available")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen/Qwen3-0.6B not available: {exc}")

    from rllm.parser import QwenChatTemplateParser

    return QwenChatTemplateParser(tokenizer), tokenizer


def _render_asst_body(parser, asst_msg: dict) -> list[int]:
    """Render an assistant message body (no header, no trailing newline)."""
    full = parser.parse_assistant(asst_msg, accumulate_reasoning=True)
    assert full.startswith(parser.assistant_token)
    body_with_eot = full[len(parser.assistant_token) :]
    # eot_token = "<|im_end|>\n"; model would stop at "<|im_end|>".
    body = body_with_eot[: -len("\n")]
    return parser.tokenizer.encode(body, add_special_tokens=False)


def test_math_tool_workflow_two_turn_canned_path():
    """End-to-end stub test: 2-turn rollout with tool call then final answer."""
    parser, tokenizer = _load_parser_and_tokenizer()

    asst_tool_call_msg = {
        "role": "assistant",
        "content": "Let me compute 12 * 7 + 3 using the calculator.",
        "reasoning": "The user asked for a simple expression; using the tool to be safe.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "12 * 7 + 3"}',
                },
            }
        ],
    }
    asst_final_msg = {
        "role": "assistant",
        "content": "The answer is \\boxed{87}.",
        "reasoning": "The tool returned 87, so the boxed answer is 87.",
    }
    canned = [
        _render_asst_body(parser, asst_tool_call_msg),
        _render_asst_body(parser, asst_final_msg),
    ]

    engine = _StubRolloutEngine(tokenizer=tokenizer, chat_parser=parser, canned_completions=canned)

    from cookbooks.math_tool_agent.workflow import MathToolWorkflow

    executor = ThreadPoolExecutor(max_workers=2)
    workflow = MathToolWorkflow(rollout_engine=engine, executor=executor, max_turns=5)
    workflow.reset(task={"instruction": "What is 12 * 7 + 3?", "answer": "87"}, uid="test")

    asyncio.run(workflow.run(task={"instruction": "What is 12 * 7 + 3?", "answer": "87"}, uid="test"))

    # Collect trajectories and verify.
    episode = workflow.collect_trajectories()
    assert len(episode.trajectories) == 1
    traj = episode.trajectories[0]
    assert traj.name == "solver"
    assert len(traj.steps) == 2, f"expected 2 steps, got {len(traj.steps)}"

    # First step: tool call; second step: final answer with reward 1.0
    assert traj.steps[-1].reward == 1.0, f"expected reward=1.0 on final step (boxed answer matches), got {traj.steps[-1].reward}"
    assert traj.steps[-1].done is True

    # Each step should have non-empty prompt_ids and response_ids.
    for i, step in enumerate(traj.steps):
        assert len(step.prompt_ids) > 0, f"step {i} has empty prompt_ids"
        assert len(step.response_ids) > 0, f"step {i} has empty response_ids"

    # Step-merging witness: turn 2's prompt_ids should start with turn 1's prompt_ids
    # PLUS turn 1's response_ids. (This is the very property the fixed completer ensures.)
    s1, s2 = traj.steps[0], traj.steps[1]
    expected_prefix = list(s1.prompt_ids) + list(s1.response_ids)
    assert list(s2.prompt_ids[: len(expected_prefix)]) == expected_prefix, "turn 2 prompt_ids does not extend turn 1's prompt_ids + response_ids — the completer's accumulation broke."


def test_math_tool_workflow_symbolic_grading():
    """LaTeX-formatted boxed answer should grade as equivalent to the
    canonical-form ground truth. Plain string-equals would miss this; the
    workflow uses ``grade_answer_mathd`` / ``grade_answer_sympy``."""
    parser, tokenizer = _load_parser_and_tokenizer()

    asst_tool_call = {
        "role": "assistant",
        "content": "Computing.",
        "reasoning": "1/2 = 0.5",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "1/2"}',
                },
            }
        ],
    }
    # Model emits the answer as a LaTeX fraction; ground truth is decimal.
    asst_final_latex = {
        "role": "assistant",
        "content": "The answer is \\boxed{\\frac{1}{2}}.",
    }
    canned = [
        _render_asst_body(parser, asst_tool_call),
        _render_asst_body(parser, asst_final_latex),
    ]
    engine = _StubRolloutEngine(tokenizer=tokenizer, chat_parser=parser, canned_completions=canned)

    from cookbooks.math_tool_agent.workflow import MathToolWorkflow

    executor = ThreadPoolExecutor(max_workers=2)
    workflow = MathToolWorkflow(rollout_engine=engine, executor=executor, max_turns=5)
    workflow.reset(task={"instruction": "What is 1/2?", "answer": "0.5"}, uid="test-latex")
    asyncio.run(workflow.run(task={"instruction": "What is 1/2?", "answer": "0.5"}, uid="test-latex"))
    episode = workflow.collect_trajectories()
    traj = episode.trajectories[0]
    assert traj.steps[-1].reward == 1.0, f"expected reward=1.0 (LaTeX \\frac{{1}}{{2}} == ground truth 0.5 symbolically), got {traj.steps[-1].reward}"


def test_math_tool_workflow_reads_math500_shape_task():
    """math500/deepscaler dataset entries use 'question' + 'ground_truth',
    not 'instruction' + 'answer'. Regression test for the field-name shim
    in ``_extract_task_fields``.
    """
    parser, tokenizer = _load_parser_and_tokenizer()

    asst_tool_call = {
        "role": "assistant",
        "content": "Computing.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "84 + 3"}',
                },
            }
        ],
    }
    asst_final = {
        "role": "assistant",
        "content": "The answer is \\boxed{87}.",
    }
    canned = [
        _render_asst_body(parser, asst_tool_call),
        _render_asst_body(parser, asst_final),
    ]
    engine = _StubRolloutEngine(tokenizer=tokenizer, chat_parser=parser, canned_completions=canned)

    from cookbooks.math_tool_agent.workflow import MathToolWorkflow

    # Use the math500-style key names: 'question' + 'ground_truth' (NOT instruction/answer).
    task = {"question": "What is 12*7+3?", "ground_truth": "87", "data_source": "math500"}
    executor = ThreadPoolExecutor(max_workers=2)
    workflow = MathToolWorkflow(rollout_engine=engine, executor=executor, max_turns=5)
    workflow.reset(task=task, uid="test-math500-shape")
    asyncio.run(workflow.run(task=task, uid="test-math500-shape"))

    episode = workflow.collect_trajectories()
    traj = episode.trajectories[0]
    assert traj.steps[-1].reward == 1.0, (
        f"workflow failed to read math500-style fields ('question'/'ground_truth'). Got reward={traj.steps[-1].reward} — expected 1.0 since the answer matches the ground truth."
    )


def test_math_tool_workflow_commits_partial_trajectory_on_max_turns():
    """When the loop hits max_turns without a final no-tool-calls turn, the
    workflow should STILL commit the partial trajectory (with whatever reward
    can be extracted from the last response). Pre-fix, the MAX_TURNS_EXCEEDED
    raise happened before commit() and the entire rollout was silently dropped.
    """
    parser, tokenizer = _load_parser_and_tokenizer()

    asst_keeps_calling = {
        "role": "assistant",
        "content": "Need another computation.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "1+1"}',
                },
            }
        ],
    }
    # Feed enough canned turns that max_turns=2 will fall through without break.
    canned = [
        _render_asst_body(parser, asst_keeps_calling),
        _render_asst_body(parser, asst_keeps_calling),
    ]
    engine = _StubRolloutEngine(tokenizer=tokenizer, chat_parser=parser, canned_completions=canned)

    from cookbooks.math_tool_agent.workflow import MathToolWorkflow

    executor = ThreadPoolExecutor(max_workers=2)
    workflow = MathToolWorkflow(rollout_engine=engine, executor=executor, max_turns=2)
    workflow.reset(task={"question": "Q?", "ground_truth": "0"}, uid="test-max-turns")

    # run() should raise TerminationEvent(MAX_TURNS_EXCEEDED), and the wrapper
    # would normally catch it. Here we use run_with_termination_handling so
    # the lifecycle matches what UnifiedWorkflowEngine does.
    asyncio.run(workflow.run_with_termination_handling(task={"question": "Q?", "ground_truth": "0"}, uid="test-max-turns"))

    episode = workflow.collect_trajectories()
    assert len(episode.trajectories) == 1, f"max_turns path silently dropped trajectory: got {len(episode.trajectories)} trajectories"
    traj = episode.trajectories[0]
    assert len(traj.steps) == 2, f"expected both turns captured, got {len(traj.steps)}"
    # Both completions emitted tool_calls and never produced \boxed{...}; reward should be 0.
    assert traj.steps[-1].reward == 0.0


def test_math_tool_workflow_wrong_answer_gets_zero_reward():
    parser, tokenizer = _load_parser_and_tokenizer()

    asst_wrong = {
        "role": "assistant",
        "content": "After computing, the answer is \\boxed{99}.",
        "reasoning": "I'll just guess.",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": '{"expression": "0"}',
                },
            }
        ],
    }
    asst_final_wrong = {
        "role": "assistant",
        "content": "Final answer: \\boxed{99}",
    }
    canned = [
        _render_asst_body(parser, asst_wrong),
        _render_asst_body(parser, asst_final_wrong),
    ]
    engine = _StubRolloutEngine(tokenizer=tokenizer, chat_parser=parser, canned_completions=canned)

    from cookbooks.math_tool_agent.workflow import MathToolWorkflow

    executor = ThreadPoolExecutor(max_workers=2)
    workflow = MathToolWorkflow(rollout_engine=engine, executor=executor, max_turns=5)
    workflow.reset(task={"instruction": "What is 12 * 7 + 3?", "answer": "87"}, uid="test-wrong")
    asyncio.run(workflow.run(task={"instruction": "What is 12 * 7 + 3?", "answer": "87"}, uid="test-wrong"))
    episode = workflow.collect_trajectories()
    assert len(episode.trajectories) == 1
    traj = episode.trajectories[0]
    assert traj.steps[-1].reward == 0.0
