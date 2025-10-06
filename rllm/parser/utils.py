PARSER_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about Python."},
    {"role": "assistant", "content": "I'll search for that.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Python programming"}'}}]},
    # {"role": "tool", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What about Java?"},
    {"role": "assistant", "content": "Let me search for Java information.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Java programming"}'}}]},
]


def fix_pad_token(tokenizer):
    """Fix pad_token if it's the same as eos_token.

    This is important because having pad_token == eos_token can cause issues during training
    where the model might learn to ignore the eos_token if it sees it used for padding.

    Args:
        tokenizer: The tokenizer to fix

    Returns:
        None (modifies tokenizer in place)
    """

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        # Try to use unk_token first, otherwise use bos_token
        if tokenizer.unk_token is not None and tokenizer.unk_token != tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.unk_token
            print(f"Set pad_token to unk_token: {tokenizer.pad_token}")
        elif tokenizer.bos_token is not None and tokenizer.bos_token != tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.bos_token
            print(f"Set pad_token to bos_token: {tokenizer.pad_token}")
        else:
            # Add a new special token for padding
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            print(f"Added new pad_token: {tokenizer.pad_token}")

    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id and eos_token_id are the same, eos_token will get masked out during training and could impact the model's performance"
