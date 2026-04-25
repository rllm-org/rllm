from __future__ import annotations

import asyncio
from types import SimpleNamespace

from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.early_finalize import maybe_generate_with_early_finalize


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        return "".join(chr(token) for token in ids)


class FakeParser:
    def __init__(self, tokenizer: FakeTokenizer):
        self.tokenizer = tokenizer

    def parse_completion(self, completion_ids: list[int]) -> dict:
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        if text.startswith("<think>") and "</think>" in text:
            reasoning, _, content = text.partition("</think>")
            return {
                "reasoning": reasoning[len("<think>") :].strip(),
                "content": content.strip(),
                "tool_calls": [],
            }
        if text.startswith("<think>"):
            return {
                "reasoning": text[len("<think>") :].strip(),
                "content": "",
                "tool_calls": [],
            }
        return {"reasoning": "", "content": text.strip(), "tool_calls": []}


class FakeWorkflow:
    def __init__(self, phase1_output: ModelOutput, phase2_output: ModelOutput | None, reserve_tokens: int = 12):
        tokenizer = FakeTokenizer()
        self.rollout_engine = SimpleNamespace(
            supports_token_in_token_out=True,
            tokenizer=tokenizer,
            chat_parser=FakeParser(tokenizer),
            max_response_length=64,
            config=SimpleNamespace(
                rllm=SimpleNamespace(
                    early_finalize=SimpleNamespace(
                        enable=True,
                        reserve_response_tokens=reserve_tokens,
                        min_phase2_tokens=2,
                        suffix_mode="auto",
                    )
                )
            ),
        )
        self.phase1_output = phase1_output
        self.phase2_output = phase2_output
        self.phase1_kwargs: dict | None = None
        self.phase2_kwargs: dict | None = None
        self.phase2_token_input: list[int] | None = None

    async def timed_llm_call(self, messages, **kwargs):  # noqa: ARG002
        self.phase1_kwargs = kwargs
        return self.phase1_output

    async def timed_llm_call_from_token_input(self, token_input, **kwargs):
        self.phase2_token_input = list(token_input)
        self.phase2_kwargs = kwargs
        assert self.phase2_output is not None
        return self.phase2_output


def _make_output(tokenizer: FakeTokenizer, prompt_text: str, completion_text: str, *, finish_reason: str) -> ModelOutput:
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    return ModelOutput(
        text=completion_text,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=[-0.1] * len(completion_ids),
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        finish_reason=finish_reason,
    )


def test_early_finalize_skips_when_phase1_budget_would_be_non_positive():
    tokenizer = FakeTokenizer()
    phase1 = _make_output(tokenizer, "prompt", "<think>abc", finish_reason="length")
    workflow = FakeWorkflow(phase1, None, reserve_tokens=24)

    result = asyncio.run(
        maybe_generate_with_early_finalize(
            workflow,
            [{"role": "user", "content": "question"}],
            application_id="task:0",
            task={"question": "question"},
            max_tokens=20,
        )
    )

    assert workflow.phase1_kwargs is not None
    assert workflow.phase1_kwargs["max_tokens"] == 20
    assert workflow.phase2_token_input is None
    assert result.output is phase1


def test_early_finalize_success_uses_reserved_tail_budget():
    tokenizer = FakeTokenizer()
    phase1 = _make_output(tokenizer, "prompt", "<think>abc", finish_reason="length")
    phase2 = _make_output(tokenizer, "unused", "42", finish_reason="stop")
    workflow = FakeWorkflow(phase1, phase2, reserve_tokens=30)

    result = asyncio.run(
        maybe_generate_with_early_finalize(
            workflow,
            [{"role": "user", "content": "question"}],
            application_id="task:0",
            task={"question": "question"},
            max_tokens=40,
        )
    )

    suffix = "</think>\nThe answer is: "
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    assert workflow.phase1_kwargs is not None
    assert workflow.phase1_kwargs["max_tokens"] == 10
    assert workflow.phase2_token_input == phase1.prompt_ids + phase1.completion_ids + suffix_ids
    assert workflow.phase2_kwargs is not None
    assert workflow.phase2_kwargs["max_tokens"] == 30 - len(suffix_ids)
    assert result.output.completion_ids == phase1.completion_ids + suffix_ids + phase2.completion_ids
    assert result.response_mask == ([1.0] * len(phase1.completion_ids) + [0.0] * len(suffix_ids) + [1.0] * len(phase2.completion_ids))
    assert result.output.content == "The answer is: 42"
    assert result.metadata is not None
    assert result.metadata["attempted"] is True


def test_early_finalize_skips_when_suffix_leaves_no_phase2_budget():
    tokenizer = FakeTokenizer()
    phase1 = _make_output(tokenizer, "prompt", "<think>abc", finish_reason="length")
    workflow = FakeWorkflow(phase1, None, reserve_tokens=4)

    result = asyncio.run(
        maybe_generate_with_early_finalize(
            workflow,
            [{"role": "user", "content": "question"}],
            application_id="task:0",
            task={"question": "question"},
            max_tokens=40,
        )
    )

    assert workflow.phase2_token_input is None
    assert result.output is phase1
    assert result.response_mask is None


def test_early_finalize_continues_without_synthetic_suffix_for_non_thinking_output():
    tokenizer = FakeTokenizer()
    phase1 = _make_output(tokenizer, "prompt", "Partial answer", finish_reason="length")
    phase2 = _make_output(tokenizer, "unused", " continued", finish_reason="stop")
    workflow = FakeWorkflow(phase1, phase2, reserve_tokens=8)

    result = asyncio.run(
        maybe_generate_with_early_finalize(
            workflow,
            [{"role": "user", "content": "question"}],
            application_id="task:0",
            task={"question": "question"},
            max_tokens=20,
        )
    )

    assert workflow.phase1_kwargs is not None
    assert workflow.phase1_kwargs["max_tokens"] == 12
    assert workflow.phase2_token_input == phase1.prompt_ids + phase1.completion_ids
    assert workflow.phase2_kwargs is not None
    assert workflow.phase2_kwargs["max_tokens"] == 8
    assert result.output.completion_ids == phase1.completion_ids + phase2.completion_ids
    assert result.response_mask == ([1.0] * len(phase1.completion_ids) + [1.0] * len(phase2.completion_ids))
    assert result.output.content == "Partial answer continued"
    assert result.metadata is not None
    assert result.metadata["suffix"] == ""
    assert result.metadata["suffix_tokens"] == 0


def test_early_finalize_allows_custom_builder_to_return_empty_suffix():
    tokenizer = FakeTokenizer()
    phase1 = _make_output(tokenizer, "prompt", "<think>abc", finish_reason="length")
    phase2 = _make_output(tokenizer, "unused", "42", finish_reason="stop")
    workflow = FakeWorkflow(phase1, phase2, reserve_tokens=8)
    workflow.build_early_finalize_suffix = lambda *args, **kwargs: ""

    result = asyncio.run(
        maybe_generate_with_early_finalize(
            workflow,
            [{"role": "user", "content": "question"}],
            application_id="task:0",
            task={"question": "question"},
            max_tokens=20,
        )
    )

    assert workflow.phase2_token_input == phase1.prompt_ids + phase1.completion_ids
    assert workflow.phase2_kwargs is not None
    assert workflow.phase2_kwargs["max_tokens"] == 8
    assert result.response_mask == ([1.0] * len(phase1.completion_ids) + [1.0] * len(phase2.completion_ids))
    assert result.metadata is not None
    assert result.metadata["suffix"] == ""


def test_workflow_local_early_finalize_config_takes_priority_over_rollout_config():
    tokenizer = FakeTokenizer()
    phase1 = _make_output(tokenizer, "prompt", "<think>abc", finish_reason="length")
    phase2 = _make_output(tokenizer, "unused", "42", finish_reason="stop")
    workflow = FakeWorkflow(phase1, phase2, reserve_tokens=8)
    workflow.early_finalize_config = SimpleNamespace(
        enable=True,
        reserve_response_tokens=30,
        min_phase2_tokens=2,
        suffix_mode="auto",
    )

    result = asyncio.run(
        maybe_generate_with_early_finalize(
            workflow,
            [{"role": "user", "content": "question"}],
            application_id="task:0",
            task={"question": "question"},
            max_tokens=40,
        )
    )

    suffix_ids = tokenizer.encode("</think>\nThe answer is: ", add_special_tokens=False)
    assert workflow.phase1_kwargs is not None
    assert workflow.phase1_kwargs["max_tokens"] == 10
    assert workflow.phase2_kwargs is not None
    assert workflow.phase2_kwargs["max_tokens"] == 30 - len(suffix_ids)
    assert result.metadata is not None


def test_early_finalize_drops_merged_logprobs_when_segments_are_missing_them():
    tokenizer = FakeTokenizer()
    phase1 = ModelOutput(
        text="<think>abc",
        prompt_ids=tokenizer.encode("prompt", add_special_tokens=False),
        completion_ids=tokenizer.encode("<think>abc", add_special_tokens=False),
        logprobs=None,
        prompt_length=6,
        completion_length=10,
        finish_reason="length",
    )
    phase2 = ModelOutput(
        text="42",
        prompt_ids=tokenizer.encode("unused", add_special_tokens=False),
        completion_ids=tokenizer.encode("42", add_special_tokens=False),
        logprobs=None,
        prompt_length=6,
        completion_length=2,
        finish_reason="stop",
    )
    workflow = FakeWorkflow(phase1, phase2, reserve_tokens=30)

    result = asyncio.run(
        maybe_generate_with_early_finalize(
            workflow,
            [{"role": "user", "content": "question"}],
            application_id="task:0",
            task={"question": "question"},
            max_tokens=40,
        )
    )

    assert result.output.completion_ids is not None
    assert result.output.logprobs is None
    assert result.response_mask is not None
