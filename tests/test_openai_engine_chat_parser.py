from types import SimpleNamespace


class DummyTokenizer:
    def encode(self, s, add_special_tokens=False):  # noqa: ARG002
        # super naive "tokenizer": map chars to ords
        return [ord(c) for c in s]


def test_openai_engine_uses_provided_chat_parser():
    # Import inside the test so our stubs from conftest are installed first.
    from rllm.engine.rollout.openai_engine import OpenAIEngine

    tok = DummyTokenizer()
    custom_parser = SimpleNamespace(name="custom")

    engine = OpenAIEngine(model="dummy", tokenizer=tok, chat_parser=custom_parser)

    assert engine.chat_parser is custom_parser


def test_agent_execution_engine_passes_chat_parser_to_openai_engine():
    from rllm.engine.agent_execution_engine import AgentExecutionEngine

    tok = DummyTokenizer()
    custom_parser = SimpleNamespace(name="custom")

    aee = AgentExecutionEngine(
        engine_name="openai",
        tokenizer=tok,
        chat_parser=custom_parser,
        rollout_engine_args={"model": "dummy"},
        config={},
        n_parallel_agents=1,
        max_workers=1,
    )

    assert getattr(aee.rollout_engine, "chat_parser", None) is custom_parser
