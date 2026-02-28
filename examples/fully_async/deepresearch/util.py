def _message_to_dict(message):
    if message is None:
        return {"role": "assistant", "content": None, "reasoning_content": None, "tool_calls": None}

    role = getattr(message, "role", "assistant")
    content = getattr(message, "content", None)
    reasoning_content = getattr(message, "reasoning_content", None)
    tool_calls = getattr(message, "tool_calls", None)

    tool_call_dicts = None
    if tool_calls:
        tool_call_dicts = []
        for tc in tool_calls:
            if tc is None or tc.function is None:
                continue
            tool_call_dicts.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )
        if not tool_call_dicts:
            tool_call_dicts = None

    message_dict = {
        "role": role,
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_call_dicts,
    }
    return message_dict


def openai_message_to_dict(messages):
    """Convert OpenAI message format to simple dict format."""
    converted = []
    for message in messages:
        if isinstance(message, dict):
            converted.append(message)
            continue
        converted.append(_message_to_dict(message))
    return converted
