"""Tests for token_accumulator module."""

from unittest.mock import MagicMock

import pytest


def _make_mock_tokenizer():
    """Create a mock tokenizer that simulates apply_chat_template behavior.

    Uses a simple scheme:
    - Special tokens: <|im_end|>=100, <|im_start|>=101, newline=10
    - apply_chat_template with continue_final_message=True returns [101, 1, 10, 101, 2, 10, 50, 51]
    - apply_chat_template with add_generation_prompt=True returns above + bridge + content
    """
    tokenizer = MagicMock()

    def _apply_chat_template(conversation, tokenize=False, **kwargs):
        if kwargs.get("continue_final_message"):
            return [101, 1, 10, 101, 2, 10, 50, 51]

        if kwargs.get("add_generation_prompt"):
            user_content = conversation[-1]["content"]
            content_ids = [ord(c) for c in user_content[:3]]
            prefix = [101, 1, 10, 101, 2, 10, 50, 51]
            bridge = [100, 10, 101, 1, 10] + content_ids + [100, 10, 101, 2, 10]
            return prefix + bridge

        return []

    tokenizer.apply_chat_template = _apply_chat_template
    return tokenizer


class TestComputeBridge:
    def test_bridge_returns_correct_ids(self):
        from rllm_model_gateway.token_accumulator import compute_bridge

        tokenizer = _make_mock_tokenizer()
        bridge = compute_bridge(tokenizer, "hi!")
        expected = [100, 10, 101, 1, 10, ord("h"), ord("i"), ord("!"), 100, 10, 101, 2, 10]
        assert bridge == expected

    def test_bridge_empty_content(self):
        from rllm_model_gateway.token_accumulator import compute_bridge

        tokenizer = _make_mock_tokenizer()
        bridge = compute_bridge(tokenizer, "")
        expected = [100, 10, 101, 1, 10, 100, 10, 101, 2, 10]
        assert bridge == expected

    def test_bridge_raises_on_prefix_mismatch(self):
        from rllm_model_gateway.token_accumulator import compute_bridge

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda *a, **kw: ([1, 2, 3] if kw.get("continue_final_message") else [9, 9, 9, 4, 5])
        with pytest.raises(ValueError, match="prefix mismatch"):
            compute_bridge(tokenizer, "test")


class TestTokenAccumulator:
    def test_initial_state(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        assert acc.turn_count == 0
        assert acc.cumulative_ids == []

    def test_ingest_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        assert acc.turn_count == 1
        assert acc.cumulative_ids == [1, 2, 3, 10, 11]

    def test_should_rewrite_false_on_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        assert acc.should_rewrite() is False

    def test_should_rewrite_true_after_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        assert acc.should_rewrite() is True

    def test_build_prompt_ids(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        prompt_ids = acc.build_next_prompt("hello")
        bridge = [100, 10, 101, 1, 10, ord("h"), ord("e"), ord("l"), 100, 10, 101, 2, 10]
        expected = [1, 2, 3, 10, 11] + bridge
        assert prompt_ids == expected

    def test_ingest_second_turn_extends_cumulative(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        # Build prompt for turn 2 (this extends cumulative_ids with bridge)
        prompt_ids = acc.build_next_prompt("hi!")
        # Now ingest turn 2's completion
        acc.ingest_turn(prompt_token_ids=prompt_ids, completion_token_ids=[20, 21])
        assert acc.turn_count == 2
        assert acc.cumulative_ids == prompt_ids + [20, 21]


class TestCumulativeVerification:
    def test_is_cumulative_true_on_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        messages = [{"role": "user", "content": "Hello"}]
        assert acc.is_cumulative(messages) is True

    def test_is_cumulative_true_when_prefix_matches(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        messages_t1 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        acc.ingest_turn([1, 2, 3], [10, 11])
        acc.update_prefix(messages_t1)

        messages_t2 = messages_t1 + [{"role": "user", "content": "How are you?"}]
        assert acc.is_cumulative(messages_t2) is True

    def test_is_cumulative_false_when_prefix_changes(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        messages_t1 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        acc.ingest_turn([1, 2, 3], [10, 11])
        acc.update_prefix(messages_t1)

        # Divergent: earlier message content changed
        messages_t2 = [
            {"role": "user", "content": "Different start"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        assert acc.is_cumulative(messages_t2) is False

    def test_is_cumulative_false_when_message_count_shrinks(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        messages_t1 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        acc.ingest_turn([1, 2, 3], [10, 11])
        acc.update_prefix(messages_t1)

        # Fewer messages than what was ingested
        messages_t2 = [{"role": "user", "content": "Fresh start"}]
        assert acc.is_cumulative(messages_t2) is False

    def test_reset_clears_state(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        acc.ingest_turn([1, 2, 3], [10, 11])
        acc.update_prefix([{"role": "user", "content": "Hello"}])
        assert acc.turn_count == 1

        acc.reset()
        assert acc.turn_count == 0
        assert acc.cumulative_ids == []
        assert acc.message_count == 0
        assert acc.should_rewrite() is False

    def test_reset_allows_fresh_ingestion(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(tokenizer=_make_mock_tokenizer())
        acc.ingest_turn([1, 2, 3], [10, 11])
        acc.update_prefix([{"role": "user", "content": "Hello"}])

        acc.reset()
        acc.ingest_turn([5, 6, 7], [20, 21])
        assert acc.turn_count == 1
        assert acc.cumulative_ids == [5, 6, 7, 20, 21]


class TestExtractNewUserContent:
    def test_extract_last_user_message(self):
        from rllm_model_gateway.token_accumulator import extract_new_user_content

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        assert extract_new_user_content(messages, prev_message_count=3) == "How are you?"

    def test_extract_handles_no_new_messages(self):
        from rllm_model_gateway.token_accumulator import extract_new_user_content

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = extract_new_user_content(messages, prev_message_count=2)
        assert result is None

    def test_extract_with_multiple_new_messages(self):
        from rllm_model_gateway.token_accumulator import extract_new_user_content

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Second question"},
        ]
        result = extract_new_user_content(messages, prev_message_count=1)
        assert result == "Second question"
