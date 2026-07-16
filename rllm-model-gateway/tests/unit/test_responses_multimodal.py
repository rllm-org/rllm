"""Unit tests for input_image handling in _input_to_messages()."""

from rllm_model_gateway._responses_compat import _input_to_messages


class TestInputImageHandling:
    """Verify input_image items are translated to Chat Completions image_url content parts."""

    def test_single_image_before_user_message(self):
        """input_image + message → merged multimodal user message."""
        input_items = [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Describe this."}]},
        ]
        system_parts, messages, images = _input_to_messages(input_items)
        assert system_parts == []
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        assert msg["content"][1] == {"type": "text", "text": "Describe this."}
        assert images == ["data:image/png;base64,AAAA"]

    def test_multiple_images_before_user_message(self):
        """Multiple input_image + message → single user message with all images + text."""
        input_items = [
            {"type": "input_image", "image_url": "data:image/png;base64,IMG1"},
            {"type": "input_image", "image_url": "data:image/png;base64,IMG2"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Compare these."}]},
        ]
        _, messages, images = _input_to_messages(input_items)
        assert len(messages) == 1
        parts = messages[0]["content"]
        assert len(parts) == 3
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "data:image/png;base64,IMG1"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,IMG2"
        assert parts[2] == {"type": "text", "text": "Compare these."}
        assert images == ["data:image/png;base64,IMG1", "data:image/png;base64,IMG2"]

    def test_image_without_following_message(self):
        """Standalone input_image at end of input → flushed as image-only user message."""
        input_items = [
            {"type": "input_image", "image_url": "data:image/png;base64,SOLO"},
        ]
        _, messages, _ = _input_to_messages(input_items)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,SOLO"}}]

    def test_image_before_string_item(self):
        """input_image + bare string → merged multimodal user message."""
        input_items = [
            {"type": "input_image", "image_url": "data:image/png;base64,X"},
            "What is this?",
        ]
        _, messages, _ = _input_to_messages(input_items)
        assert len(messages) == 1
        parts = messages[0]["content"]
        assert parts[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}}
        assert parts[1] == {"type": "text", "text": "What is this?"}

    def test_image_flushed_before_function_call(self):
        """input_image is flushed as standalone message before function_call items."""
        input_items = [
            {"type": "input_image", "image_url": "data:image/png;base64,PRE"},
            {"type": "function_call", "name": "calc", "arguments": "{}", "call_id": "c1"},
        ]
        _, messages, _ = _input_to_messages(input_items)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,PRE"}}]
        assert messages[1]["role"] == "assistant"

    def test_no_image_preserves_original_behavior(self):
        """Without input_image, behavior is unchanged (plain string content)."""
        input_items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
        ]
        _, messages, images = _input_to_messages(input_items)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert images == []

    def test_codex_cli_typical_payload(self):
        """Simulate realistic Codex CLI --image payload."""
        input_items = [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANS"},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is the value of bar C in this chart? Print only the number."}],
            },
        ]
        system_parts, messages, _ = _input_to_messages(input_items)
        assert system_parts == []
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image_url"
        assert msg["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert msg["content"][1]["type"] == "text"
        assert "bar C" in msg["content"][1]["text"]

    def test_image_between_conversations(self):
        """Image in a multi-turn conversation merges with the next user message only."""
        input_items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "First turn"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Reply 1"}]},
            {"type": "input_image", "image_url": "data:image/png;base64,TURN2IMG"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Now look at this image"}]},
        ]
        _, messages, _ = _input_to_messages(input_items)
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "First turn"}
        assert messages[1] == {"role": "assistant", "content": "Reply 1"}
        assert messages[2]["role"] == "user"
        assert isinstance(messages[2]["content"], list)
        assert len(messages[2]["content"]) == 2
        assert messages[2]["content"][0]["type"] == "image_url"
        assert messages[2]["content"][1] == {"type": "text", "text": "Now look at this image"}
