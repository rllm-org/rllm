"""A renderer stand-in for cumulative token mode, which loads no model."""


class MockRendered:
    """Stand-in for renderers.RenderedTokens (only .token_ids is consumed)."""

    def __init__(self, token_ids):
        self.token_ids = token_ids


class MockRenderer:
    """Mimics renderers.Renderer.bridge_to_next_turn without loading a real model.

    Bridge output = prev_prompt + prev_completion + deterministic extension:
        [100, 10, 101, 1, 10] + content_ids + [100, 10, 101, 2, 10]
    where content_ids = ord() of the first 3 chars of the last user message.

    Returns None when new_messages is empty or contains an assistant turn —
    mirroring renderers' reject_assistant_in_extension contract.
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
        return MockRendered(list(prev_prompt_ids) + list(prev_completion_ids) + bridge)
