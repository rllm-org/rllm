PARSER_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about Python."},
    {"role": "assistant", "content": "I'll search for that.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Python programming"}'}}]},
    # {"role": "tool", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What about Java?"},
    {"role": "assistant", "content": "Let me search for Java information.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Java programming"}'}}]},
]

# Simple multi-turn messages for verify_equivalence tests.
# Ends with a user message (representing the prompt before model generation)
# to avoid HF template quirks like Qwen3's <think> tag insertion on the last
# assistant message after the last user query.
SIMPLE_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
    {"role": "user", "content": "What is the capital of France?"},
]
