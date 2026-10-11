import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rllm.types import Step

# Load the helper without importing optional training backends.
_PATH = Path(__file__).resolve().parents[2] / "rllm/trainer/distill/advantage.py"
_spec = importlib.util.spec_from_file_location("rllm_distill_advantage_test", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("privileged_prompt", [False, True])
@pytest.mark.parametrize("scored_tokens", [0, 1, 3])
def test_unscored_teacher_suffix_has_zero_advantage(privileged_prompt, scored_tokens):
    step = Step(
        prompt_ids=[10, 11],
        response_ids=[20, 21, 22],
        logprobs=[-1.0, -2.0, -3.0],
        chat_completions=[{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
    )
    prompt_ids = [30, 31, 32] if privileged_prompt else step.prompt_ids
    teacher_scores = [-0.5, -2.5, -2.0][:scored_tokens]
    # TinkerEngine.compute_logprobs truncates its input to max_model_length.
    teacher = SimpleNamespace(compute_logprobs=AsyncMock(return_value=[0.0] * len(prompt_ids) + teacher_scores))
    kwargs = {}
    if privileged_prompt:
        kwargs = {
            "teacher_prompt_fn": lambda messages: messages,
            "teacher_chat_parser": SimpleNamespace(parse=Mock(return_value="teacher prompt")),
            "teacher_tokenizer": SimpleNamespace(encode=Mock(return_value=prompt_ids)),
        }

    advantages = asyncio.run(_module.compute_step_distill_advantage(step, teacher, shared_tokenizer=True, **kwargs))

    expected = [0.5, -0.5, 1.0][:scored_tokens] + [0.0] * (3 - scored_tokens)
    assert advantages == pytest.approx(expected)
    assert len(advantages) == len(step.response_ids)
    teacher.compute_logprobs.assert_awaited_once_with(prompt_ids + step.response_ids)
    assert step.logprobs == [-1.0, -2.0, -3.0]


def test_missing_teacher_scores_do_not_leak_into_discounted_prefix():
    step = Step(prompt_ids=[10], response_ids=[20, 21, 22], logprobs=[-1.0, -2.0, -3.0])
    teacher = SimpleNamespace(compute_logprobs=AsyncMock(return_value=[0.0, -0.5, -2.5]))

    advantages = asyncio.run(_module.compute_step_distill_advantage(step, teacher, shared_tokenizer=True, kl_penalty_coef=2.0, kl_discount_factor=0.5, clip_min=-0.75, clip_max=0.75))

    assert advantages == pytest.approx([0.5, -0.75, 0.0])
