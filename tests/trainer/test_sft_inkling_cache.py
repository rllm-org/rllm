"""Inkling preflight reuses BPE work without caching validation results."""

from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")
InklingRenderer = pytest.importorskip("renderers.inkling").InklingRenderer

from rllm.trainer.sft import tinker_dataset as td  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402


class _Tokenizer:
    """Offline character tokenizer; exercise the real Inkling renderer."""

    unk_token_id = -1

    def __init__(self):
        self.specials = {}

    def convert_tokens_to_ids(self, token):
        return self.specials.setdefault(token, 1_000_000 + len(self.specials))

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


@pytest.fixture
def renderer():
    return InklingRenderer(_Tokenizer())


def _messages():
    return [
        {"role": "user", "content": "Inspect the files."},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "First inspect the directory."}],
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "one.py"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "There is one file."}, {"type": "text", "text": "Found one.py."}]},
        {"role": "user", "content": "Confirm completion."},
        {"role": "assistant", "content": "Done."},
    ]


@pytest.mark.parametrize("last_only", [False, True])
@pytest.mark.parametrize("loss_reduction", ["none", "sequence_mean"])
def test_preflight_preserves_datums_and_releases_cache_between_rows(renderer, monkeypatch, last_only, loss_reduction):
    messages = _messages()
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    kwargs = {"last_only": last_only, "tools": tools, "loss_reduction": loss_reduction}
    expected = td.conversation_to_datum(messages, renderer, 262144, **kwargs)
    calls = Counter()
    original = renderer._encode

    def encode(text):
        calls[text] += 1
        return original(text)

    monkeypatch.setattr(renderer, "_encode", encode)
    for invocation in range(1, 3):
        actual = td.conversation_to_datum(messages, renderer, 262144, validate_prefix_stability=True, **kwargs)
        assert actual.model_input.to_ints() == expected.model_input.to_ints()
        for field in ("target_tokens", "weights"):
            assert actual.loss_fn_inputs[field].data == expected.loss_fn_inputs[field].data
        assert set(calls.values()) == {invocation}, "each text is encoded once per row, with a fresh cache on the next call"
        assert renderer._encode is encode, "the shared training renderer must remain untouched"


@pytest.mark.parametrize("broken_prefix", [2, 4])
def test_preflight_still_rejects_each_rewritten_prefix(renderer, monkeypatch, broken_prefix):
    original = InklingRenderer.render
    seen = []

    def render(self, messages, **kwargs):
        seen.append(len(messages))
        rendered = original(self, messages, **kwargs)
        if len(messages) == broken_prefix:
            rendered.token_ids[0] += 1
        return rendered

    monkeypatch.setattr(InklingRenderer, "render", render)
    with pytest.raises(SFTConfigError, match=f"rendered prefix through trainable message {broken_prefix - 1} is rewritten"):
        td.conversation_to_datum(_messages(), renderer, 262144, validate_prefix_stability=True)
    assert seen == [6, *range(2, broken_prefix + 1, 2)]


def test_cache_uses_exact_text_and_evicts_beyond_its_bound(renderer, monkeypatch):
    calls = Counter()
    original = renderer._encode

    def encode(text):
        calls[text] += 1
        return original(text)

    monkeypatch.setattr(renderer, "_encode", encode)
    cached = td._preflight_renderer(renderer)
    assert cached._encode("same") == cached._encode("same")
    assert cached._encode("same ") != cached._encode("same")
    assert calls == {"same": 1, "same ": 1}
    for index in range(4096):
        cached._encode(str(index))
    assert cached._encode.cache_info().currsize == 4096
    cached._encode("same")
    assert calls["same"] == 2
    td._preflight_renderer(renderer)._encode("same")
    assert calls["same"] == 3


def test_other_renderers_and_inkling_subclasses_are_not_cached():
    class ContextDependentInkling(InklingRenderer):
        pass

    for renderer in (object(), ContextDependentInkling(_Tokenizer())):
        assert td._preflight_renderer(renderer) is renderer
