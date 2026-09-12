"""Offline judge/adapter tests. No external API or sandbox calls."""

import asyncio
import copy
import json

import httpx
import pytest
from omegaconf import OmegaConf

from rllm.rewards.cheating_judge import CheatingJudge, CheatingJudgeConfig, build_cheating_judge, episode_to_judge_input, parse_verdict
from rllm.rewards.cheating_prompt import format_user_message
from rllm.trainer.algorithms.config import CompactFilteringConfig
from rllm.trainer.buffer import TrajectoryGroupBuffer
from rllm.trainer.metrics_aggregator import MetricsAggregator
from rllm.types import Episode, Step, StepDelta, Task, TerminationReason, Trajectory, TrajectoryDelta


def completion(content="### PASS", **fields):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": content, **fields}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "completion_tokens_details": {"reasoning_tokens": 5}},
    }


def agent(command="echo hello", call_id="a", **fields):
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}], **fields}


def episode(reward=1.0, *, compact=False):
    first = [{"role": "user", "content": "Solve this task."}, agent(reasoning_content="Read the input.", content="Checking.")]
    suffix = [{"role": "tool", "tool_call_id": "a", "content": json.dumps({"returncode": 0, "output": "hello\n"})}, agent("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "b")]
    if compact:
        steps = [
            StepDelta(id="s1", parent_step_id=None, prompt_ids_suffix=[1], response_ids=[2], logprobs=[0.0], chat_completions_suffix=first),
            StepDelta(id="s2", parent_step_id="s1", prompt_ids_suffix=[3], response_ids=[4], logprobs=[0.0], chat_completions_suffix=suffix),
        ]
        trajectory = TrajectoryDelta(name="solver", steps=steps, reward=reward, signals={"accuracy": reward})
    else:
        trajectory = Trajectory(name="solver", steps=[Step(id="s1", chat_completions=first), Step(id="s2", chat_completions=first + suffix)], reward=reward, signals={"accuracy": reward})
    return Episode(id="task:0", is_correct=bool(reward), termination_reason=TerminationReason.ENV_DONE, trajectories=[trajectory], metrics={"accuracy": reward})


TASK = Task(id="task", instruction="Solve this task.")


@pytest.fixture
def make_judge(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY", "test-secret-not-for-output")

    def make(handler, **kwargs):
        config = CheatingJudgeConfig(model="frozen-judge", api_key_env="TEST_JUDGE_KEY", retry_delay_s=0, max_attempts=2, **kwargs)
        return CheatingJudge(config, transport=httpx.MockTransport(handler))

    return make


def test_compact_and_flat_prompts_identical_without_mutation():
    flat, compact = episode(), episode(compact=True)
    before = copy.deepcopy(compact)
    expected = format_user_message({"instruction": TASK.instruction, "messages": flat.trajectories[0].steps[-1].chat_completions})
    assert format_user_message(episode_to_judge_input(TASK, flat)) == expected
    assert format_user_message(episode_to_judge_input(TASK, compact)) == expected
    assert compact == before
    assert "reasoning:\nRead the input." in expected["content"]
    assert "content:\nChecking." in expected["content"]
    assert "status: no tool result recorded" not in expected["content"]


def test_multiple_calls_errors_and_missing_results_preserved():
    e = episode()
    a = agent("cat missing", "a", reasoning_content="Investigate")
    a["tool_calls"].extend(agent("echo second", "b")["tool_calls"])
    messages = [a, {"role": "tool", "tool_call_id": "b", "content": {"returncode": 1, "output_head": "failed", "elided_chars": 99, "exception_info": "timeout"}}, agent("echo final", "c")]
    e.trajectories[0].steps = [Step(chat_completions=messages)]
    prompt = format_user_message(episode_to_judge_input(TASK, e))["content"].split("<end_trajectory>")[0]
    assert "command 1:\ncat missing\ncommand 2:\necho second" in prompt
    assert "<environment command=2>\nexit_code: 1" in prompt
    assert "elided_chars:\n99" in prompt and "exception_info:\ntimeout" in prompt
    assert "<environment command=1>\nstatus: no tool result recorded before next non-tool message" in prompt
    assert "status: no tool result recorded before trajectory ended" in prompt


@pytest.mark.parametrize("kind", ["no_trajectories", "empty", "no_assistant", "bad_parent", "duplicate"])
def test_invalid_evidence_is_rejected(kind):
    e = episode(compact=kind in ("bad_parent", "duplicate"))
    if kind == "no_trajectories":
        e.trajectories = []
    elif kind == "empty":
        e.trajectories[0].steps = []
    elif kind == "no_assistant":
        e.trajectories[0].steps = [Step(chat_completions=[{"role": "user", "content": "hi"}])]
    elif kind == "bad_parent":
        e.trajectories[0].steps[-1].parent_step_id = "missing"
    else:
        e.trajectories[0].steps[-1].id = "s1"
    with pytest.raises(ValueError):
        episode_to_judge_input(TASK, e)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("later_turns", [1, 2, 3])
def test_longest_path_by_turns_with_later_tie_break_without_mutation(compact, later_turns):
    e = episode(compact=compact)
    earlier = episode_to_judge_input(TASK, e)["messages"]
    later = [agent("echo later", str(i), content="x" * 10000) for i in range(later_turns)]
    if compact:
        # A root may contain full chat history after a token-prefix reset.
        extra = StepDelta(id="reset", parent_step_id=None, prompt_ids_suffix=[], chat_completions_suffix=later)
    else:
        extra = Step(id="reset", chat_completions=later)
    e.trajectories[0].steps.append(extra)
    before = copy.deepcopy(e)
    assert episode_to_judge_input(TASK, e)["messages"] == (later if later_turns >= 2 else earlier)
    assert e == before


@pytest.mark.parametrize("layout", ["within", "across", "both"])
@pytest.mark.parametrize("reverse", [False, True])
def test_delta_longest_path_across_container_layouts(layout, reverse):
    e = episode(compact=True)
    original = e.trajectories[0]
    alternative = copy.deepcopy(original)
    alternative.steps[-1].chat_completions_suffix[-1] = agent("echo alternative", "b")
    extension = StepDelta(
        id="s3", parent_step_id="s2", prompt_ids_suffix=[5], response_ids=[6], logprobs=[0.0],
        chat_completions_suffix=[{"role": "tool", "tool_call_id": "b", "content": "ok"}, agent("echo longest", "c")],
    )
    alternative.steps.append(extension)
    expected = episode_to_judge_input(TASK, Episode(trajectories=[alternative]))["messages"]
    if layout == "within":
        # Both paths share s1, but the longer path has its own s2 sibling.
        alternative.steps[1].id = "retry"
        extension.parent_step_id = "retry"
        branches = [[original.steps[1]], alternative.steps[1:]]
        if reverse:
            branches.reverse()
        original.steps = [original.steps[0]] + [step for branch in branches for step in branch]
    else:
        if layout == "both":
            sibling = copy.deepcopy(alternative.steps[1])
            sibling.id = "short_retry"
            sibling.chat_completions_suffix[-1] = agent("echo shorter", "b")
            alternative.steps.append(sibling)
        e.trajectories = [original, alternative]
        if reverse:
            e.trajectories.reverse()
    before = copy.deepcopy(e)
    assert episode_to_judge_input(TASK, e)["messages"] == expected
    assert e == before


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
@pytest.mark.parametrize("compact", [False, True])
def test_branch_selection_preserves_all_training_steps_and_shares_penalty(make_judge, compact, verdict):
    e = episode(compact=compact)
    trajectory = e.trajectories[0]
    selected = copy.deepcopy(trajectory.steps[-1])
    selected.id = "retry"
    if compact:
        selected.chat_completions_suffix[-1] = agent("echo selected", "b")
    else:
        selected.chat_completions[-1] = agent("echo selected", "b")
    trajectory.steps.append(selected)
    other = copy.deepcopy(trajectory)
    other.steps = other.steps[:1]
    e.trajectories.append(other)
    for t in e.trajectories:
        for step in t.steps:
            step.reward = 1.0
    before = copy.deepcopy(e.trajectories)

    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        assert "echo selected" in prompt
        assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in prompt
        return httpx.Response(200, json=completion(f"### {verdict}"))

    asyncio.run(make_judge(handler).apply(TASK, e))
    assert e.metadata["cheating_judge"]["selected_num_turns"] == 2
    assert e.termination_reason == TerminationReason.ENV_DONE
    for actual, original in zip(e.trajectories, before, strict=True):
        assert actual.reward == float(verdict == "PASS")
        assert len(actual.steps) == len(original.steps)
        for step, old in zip(actual.steps, original.steps, strict=True):
            assert step.reward == float(verdict == "PASS")
            old.reward = step.reward
            assert step == old


@pytest.mark.parametrize("content,expected", [("### FAIL then ### PASS", "PASS"), ("### PASS\n### FAIL", "FAIL")])
def test_parser_uses_last_content_marker(content, expected):
    assert parse_verdict(completion(content, reasoning_content="### FAIL\n### PASS")) == expected


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"choices": []},
        completion("PASS"),
        completion("### PASSING"),
        completion(None, reasoning_content="### PASS"),
        {"choices": [{"finish_reason": "length", "message": {"content": "### PASS"}}]},
        completion("### PASS", tool_calls=[{}]),
    ],
)
def test_parser_rejects_invalid_or_incomplete_response(data):
    with pytest.raises(ValueError):
        parse_verdict(data)


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
@pytest.mark.parametrize("compact", [False, True])
def test_reward_mask_and_request_contract(make_judge, verdict, compact):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(f"Evidence.\n### {verdict}"))

    judge = make_judge(handler)
    e = episode(compact=compact)
    for step in e.trajectories[0].steps:
        step.reward = 1.0
    asyncio.run(judge.apply(TASK, e))
    assert len(requests) == 1
    assert set(requests[0]) == {"model", "messages", "reasoning_effort"}
    assert requests[0]["model"] == "frozen-judge" and requests[0]["reasoning_effort"] == "high"
    assert [m["role"] for m in requests[0]["messages"]] == ["user"]
    assert e.is_correct == (verdict == "PASS")
    assert e.trajectories[0].reward == float(verdict == "PASS")
    assert type(e.trajectories[0].reward) is float
    assert e.trajectories[0].signals["accuracy"] == float(verdict == "PASS")
    assert e.metrics["accuracy"] == float(verdict == "PASS")
    assert e.termination_reason == TerminationReason.ENV_DONE
    assert e.metrics["cheat_frac"] == float(verdict == "FAIL")
    assert e.metrics["verifier_correct_frac"] == 1.0
    assert e.metrics["judge_error_frac"] == 0.0 and e.metrics["judge_coverage"] == 1.0
    assert e.metrics["judge_prompt_tokens"] == 11.0 and e.metrics["judge_reasoning_tokens"] == 5.0
    assert e.metadata["cheating_judge"]["original_rewards"] == [1.0]
    assert "test-secret-not-for-output" not in json.dumps(e.to_dict())
    if verdict == "FAIL":
        assert all(s.reward == 0.0 for s in e.trajectories[0].steps)


@pytest.mark.parametrize("skip", ["incorrect", "validation", "infra"])
def test_skips_without_api_call(make_judge, skip):
    def handler(request):
        pytest.fail("Judge must not be called")

    e = episode(0.0 if skip == "incorrect" else 1.0)
    if skip == "infra":
        e.termination_reason = TerminationReason.VERIFIER_TIMEOUT
    before = copy.deepcopy(e)
    asyncio.run(make_judge(handler).apply(TASK, e, is_validation=skip == "validation"))
    assert e.trajectories == before.trajectories and e.is_correct == before.is_correct
    assert "cheat_frac" not in e.metrics and "judge_error_frac" not in e.metrics
    if skip == "validation":
        assert e == before


@pytest.mark.parametrize("mode", ["malformed", "timeout", "rate_limit", "server_error", "auth"])
def test_judge_failures_filter_not_cheating(make_judge, mode):
    calls = []

    def handler(request):
        calls.append(request)
        if mode == "timeout":
            raise httpx.ReadTimeout("test-secret-not-for-output", request=request)
        status = {"rate_limit": 429, "server_error": 503, "auth": 401}.get(mode, 200)
        return httpx.Response(status, json=completion("no verdict"))

    e = episode()
    asyncio.run(make_judge(handler).apply(TASK, e))
    assert len(calls) == (1 if mode == "auth" else 2)
    assert e.termination_reason == TerminationReason.GRADING_ERROR
    assert not e.is_correct and e.trajectories[0].reward == 0.0
    assert e.metrics["judge_error_frac"] == 1.0 and e.metrics["judge_coverage"] == 0.0
    assert e.metrics["reward_removed_frac"] == 0.0 and "cheat_frac" not in e.metrics
    assert e.metadata["error"]["error_type"] == "CheatingJudgeError"
    assert "test-secret-not-for-output" not in json.dumps(e.to_dict())
    assert CompactFilteringConfig(enable=True, mask_termination_reasons=["grading_error"]).should_mask(e.termination_reason)


def test_retry_recovers_and_counts_usage(make_judge):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion("malformed" if len(calls) == 1 else "### FAIL"))

    e = episode()
    asyncio.run(make_judge(handler).apply(TASK, e))
    assert len(calls) == 2 and e.metrics["judge_requests"] == 2.0
    assert e.metrics["judge_prompt_tokens"] == 22.0
    assert e.metrics["cheat_frac"] == 1.0 and e.metrics["judge_error_frac"] == 0.0


def test_invalid_correct_episode_is_grading_error(make_judge):
    e = episode()
    e.trajectories.append(copy.deepcopy(e.trajectories[0]))
    e.trajectories[-1].reward = 0.0
    asyncio.run(make_judge(lambda _: pytest.fail("No request for invalid input")).apply(TASK, e))
    assert e.termination_reason == TerminationReason.GRADING_ERROR
    assert e.metrics["judge_requests"] == 0 and "cheat_frac" not in e.metrics


def test_parallel_judging_has_no_separate_concurrency_limit(make_judge):
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return httpx.Response(200, json=completion())

    judge = make_judge(handler)

    async def run():
        await asyncio.gather(*(judge.apply(TASK, episode()) for _ in range(7)))

    asyncio.run(run())
    assert peak == 7
    asyncio.run(run())  # The same judge also works from a new caller event loop.


def test_metric_denominators_include_filtered_episodes(make_judge):
    clean, cheating, failed, incorrect = episode(), episode(), episode(), episode(0.0)
    for e, answer in [(clean, "### PASS"), (cheating, "### FAIL"), (failed, "malformed"), (incorrect, "unused")]:
        asyncio.run(make_judge(lambda _, answer=answer: httpx.Response(200, json=completion(answer))).apply(TASK, e))
    buf = TrajectoryGroupBuffer.__new__(TrajectoryGroupBuffer)
    buf._aggregator = MetricsAggregator()
    buf._record_episode_metrics([clean, cheating, failed, incorrect])
    metrics = buf._aggregator.flush()
    assert metrics["episode/cheat_frac"] == 0.5  # 1 of 2 successfully judged
    assert metrics["episode/judge_error_frac"] == pytest.approx(1 / 3)  # 1 of 3 eligible
    assert metrics["episode/judge_coverage"] == pytest.approx(2 / 3)
    assert metrics["episode/verifier_correct_frac"] == 0.75
    assert metrics["episode/reward_removed_frac"] == 0.25  # error isn't cheating
    assert metrics["episode/correct"] == 0.25
    # The cheating attempt stays ENV_DONE but is incorrect; judge errors are excluded.
    assert metrics["episode/env_done_correct_frac"] == pytest.approx(1 / 3)


def test_config_requires_error_filter_and_key(monkeypatch):
    config = OmegaConf.create({"enabled": True, "model": "judge", "api_key_env": "TEST_JUDGE_KEY"})
    with pytest.raises(ValueError, match="compact filtering"):
        build_cheating_judge(config, CompactFilteringConfig())
    monkeypatch.delenv("TEST_JUDGE_KEY", raising=False)
    filtering = CompactFilteringConfig(enable=True, mask_termination_reasons=["grading_error"])
    with pytest.raises(ValueError, match="TEST_JUDGE_KEY"):
        build_cheating_judge(config, filtering)
    assert build_cheating_judge({"enabled": False}, CompactFilteringConfig()) is None
