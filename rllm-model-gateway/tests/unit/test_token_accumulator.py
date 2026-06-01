"""Tests for token_accumulator module."""


class _MockRendered:
    """Stand-in for renderers.RenderedTokens (only .token_ids is used)."""

    def __init__(self, token_ids):
        self.token_ids = token_ids


class _MockRenderer:
    """Mimics renderers.Renderer.bridge_to_next_turn for tests.

    Bridge output = prev_prompt + prev_completion + a deterministic extension:
        [100, 10, 101, 1, 10] + content_ids + [100, 10, 101, 2, 10]
    where content_ids = ord() of the first 3 chars of the last user message.

    Returns None when any new message is an assistant turn (mirroring
    renderers' reject_assistant_in_extension) or when new_messages is empty.
    """

    def bridge_to_next_turn(self, prev_prompt_ids, prev_completion_ids, new_messages, *, tools=None):
        if not new_messages or any(m.get("role") == "assistant" for m in new_messages):
            return None
        content = ""
        for m in reversed(new_messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                break
        content_ids = [ord(c) for c in content[:3]]
        bridge = [100, 10, 101, 1, 10] + content_ids + [100, 10, 101, 2, 10]
        return _MockRendered(list(prev_prompt_ids) + list(prev_completion_ids) + bridge)


class TestTokenAccumulator:
    def test_initial_state(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        assert acc.turn_count == 0
        assert acc.cumulative_ids == []

    def test_ingest_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        assert acc.turn_count == 1
        assert acc.cumulative_ids == [1, 2, 3, 10, 11]
        assert acc.prev_prompt_ids == [1, 2, 3]
        assert acc.prev_completion_ids == [10, 11]

    def test_should_rewrite_false_on_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        assert acc.should_rewrite() is False

    def test_should_rewrite_true_after_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        assert acc.should_rewrite() is True

    def test_build_prompt_ids(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        prompt_ids = acc.build_next_prompt([{"role": "user", "content": "hello"}])
        bridge = [100, 10, 101, 1, 10, ord("h"), ord("e"), ord("l"), 100, 10, 101, 2, 10]
        expected = [1, 2, 3, 10, 11] + bridge
        assert prompt_ids == expected

    def test_build_prompt_returns_none_on_assistant_in_new_messages(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        # An assistant message in the new slice must make the bridge bail out.
        result = acc.build_next_prompt([{"role": "assistant", "content": "sampled"}, {"role": "user", "content": "x"}])
        assert result is None

    def test_ingest_second_turn_replaces_state(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn(prompt_token_ids=[1, 2, 3], completion_token_ids=[10, 11])
        # Build prompt for turn 2.
        prompt_ids = acc.build_next_prompt([{"role": "user", "content": "hi!"}])
        # Ingest turn 2's completion: prev_prompt becomes the bridge prompt,
        # prev_completion becomes the new completion.
        acc.ingest_turn(prompt_token_ids=prompt_ids, completion_token_ids=[20, 21])
        assert acc.turn_count == 2
        assert acc.prev_prompt_ids == prompt_ids
        assert acc.prev_completion_ids == [20, 21]
        assert acc.cumulative_ids == prompt_ids + [20, 21]

    def test_turn3_prompt_extends_turn2(self):
        """The bridge contract: each turn's prompt starts with the prior full sequence."""
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn([1, 2, 3], [10, 11])
        p2 = acc.build_next_prompt([{"role": "user", "content": "abc"}])
        acc.ingest_turn(p2, [20, 21])
        p3 = acc.build_next_prompt([{"role": "user", "content": "def"}])
        # p3 must start with the full turn-2 sequence (p2 + completion2).
        assert p3[: len(p2) + 2] == p2 + [20, 21]


class TestCumulativeVerification:
    def test_is_cumulative_true_on_first_turn(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        messages = [{"role": "user", "content": "Hello"}]
        assert acc.is_cumulative(messages) is True

    def test_is_cumulative_true_when_prefix_matches(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
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

        acc = TokenAccumulator(renderer=_MockRenderer())
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

        acc = TokenAccumulator(renderer=_MockRenderer())
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

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn([1, 2, 3], [10, 11])
        acc.update_prefix([{"role": "user", "content": "Hello"}])
        assert acc.turn_count == 1

        acc.reset()
        assert acc.turn_count == 0
        assert acc.cumulative_ids == []
        assert acc.prev_prompt_ids == []
        assert acc.prev_completion_ids == []
        assert acc.message_count == 0
        assert acc.should_rewrite() is False

    def test_reset_allows_fresh_ingestion(self):
        from rllm_model_gateway.token_accumulator import TokenAccumulator

        acc = TokenAccumulator(renderer=_MockRenderer())
        acc.ingest_turn([1, 2, 3], [10, 11])
        acc.update_prefix([{"role": "user", "content": "Hello"}])

        acc.reset()
        acc.ingest_turn([5, 6, 7], [20, 21])
        assert acc.turn_count == 1
        assert acc.cumulative_ids == [5, 6, 7, 20, 21]


class TestExtractNewMessages:
    def test_extract_new_user_message(self):
        from rllm_model_gateway.token_accumulator import extract_new_messages

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        # prev_message_count=3 means [system, user, assistant] already processed.
        # The new slice is [user "How are you?"]; no assistant to drop.
        assert extract_new_messages(messages, prev_message_count=3) == [{"role": "user", "content": "How are you?"}]

    def test_extract_drops_assistant_from_new_slice(self):
        from rllm_model_gateway.token_accumulator import extract_new_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Next"},
        ]
        # prev_message_count=1 → new slice is [assistant, user]; assistant dropped.
        assert extract_new_messages(messages, prev_message_count=1) == [{"role": "user", "content": "Next"}]

    def test_extract_handles_no_new_messages(self):
        from rllm_model_gateway.token_accumulator import extract_new_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert extract_new_messages(messages, prev_message_count=2) == []

    def test_extract_keeps_tool_and_user_messages(self):
        from rllm_model_gateway.token_accumulator import extract_new_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "use tool"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "thanks"},
        ]
        # prev=2 → slice [tool, user]; both kept (only assistants dropped).
        assert extract_new_messages(messages, prev_message_count=2) == [
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "thanks"},
        ]
