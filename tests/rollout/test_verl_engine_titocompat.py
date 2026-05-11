"""Regression test for the VerlEngine + TITOCompleter contract.

The completer calls ``rollout_engine.assemble_model_output(token_input, token_output)``
without ``prompt_ids`` in kwargs. ``VerlEngine.assemble_model_output`` must
return a ``ModelOutput`` with ``prompt_ids`` populated (= ``token_input``)
so the downstream Verl transform layer doesn't silently skip the step.

Pre-fix: ``prompt_ids = kwargs.pop("prompt_ids", None)`` left it as ``None``,
the transform layer at ``rllm/experimental/verl/transform.py:262`` then
warned and skipped the step, so the Verl + workflow + TITO path produced
zero training rows.

Post-fix: when kwargs is missing ``prompt_ids``, it defaults to ``token_input``.

This test instantiates a minimal ``VerlEngine`` (bypasses ``__init__`` since
that needs a real server manager + hydra config), wires only the tokenizer
and parser, and calls ``assemble_model_output`` with the two-arg signature
the completer uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _StubVerlTokenOutput:
    token_ids: list[int]
    log_probs: list[float]
    stop_reason: str = "stop"


def _make_minimal_verl_engine():
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
    except ImportError:
        pytest.skip("transformers not available")
    try:
        from rllm.experimental.rollout.verl_engine import VerlEngine  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"VerlEngine not importable: {exc}")

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen3-0.6B not available: {exc}")

    from rllm.parser import QwenChatTemplateParser

    # Bypass __init__ — it requires verl AsyncLLMServerManager + Hydra config.
    engine = object.__new__(VerlEngine)
    engine.tokenizer = tokenizer
    engine.chat_parser = QwenChatTemplateParser(tokenizer)
    return engine


def test_assemble_model_output_without_prompt_ids_kwarg_defaults_to_token_input():
    """The TITO path passes only (token_input, token_output) — no kwargs.
    The fix defaults ``prompt_ids`` to ``token_input`` in that case."""
    engine = _make_minimal_verl_engine()
    # A token_input that decodes cleanly; doesn't have to be semantically meaningful.
    token_input = engine.tokenizer.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    completion_ids = engine.tokenizer.encode("Hello.<|im_end|>", add_special_tokens=False)
    token_output = _StubVerlTokenOutput(token_ids=completion_ids, log_probs=[0.0] * len(completion_ids))

    # This is exactly how TITOCompleter calls it: no kwargs.
    model_output = engine.assemble_model_output(token_input=token_input, token_output=token_output)

    assert model_output.prompt_ids is not None, (
        "VerlEngine.assemble_model_output regression: prompt_ids is None when called "
        "without an explicit prompt_ids kwarg. The Verl transform layer will silently "
        "skip every step in this state — fix in verl_engine.py."
    )
    assert list(model_output.prompt_ids) == list(token_input)
    assert model_output.prompt_length == len(token_input)
    assert list(model_output.completion_ids) == list(completion_ids)


def test_assemble_model_output_with_explicit_prompt_ids_still_overrides():
    """The non-TITO path passes a separate ``prompt_ids`` kwarg (= the
    "logical" prompt without engine-internal prepends). That should still win."""
    engine = _make_minimal_verl_engine()
    full_input = engine.tokenizer.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    logical_prompt = full_input[:5]  # arbitrary subset to show the kwarg overrides
    completion_ids = engine.tokenizer.encode("Hello.<|im_end|>", add_special_tokens=False)
    token_output = _StubVerlTokenOutput(token_ids=completion_ids, log_probs=[])

    model_output = engine.assemble_model_output(
        token_input=full_input,
        token_output=token_output,
        prompt_ids=logical_prompt,
    )

    assert list(model_output.prompt_ids) == list(logical_prompt), "Explicit prompt_ids kwarg should override the token_input default."
    assert model_output.prompt_length == len(logical_prompt)
