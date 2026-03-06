"""Tests for scorers."""

from unittest.mock import MagicMock

from rllm_telemetry.scorers import (
    JUDGE_PROMPTS,
    Contains,
    ExactMatch,
    LlmJudge,
    Score,
    Scorer,
    _parse_judge_response,
)


class TestScore:
    def test_score_fields(self):
        s = Score(name="test", value=0.5, metadata={"key": "val"})
        assert s.name == "test"
        assert s.value == 0.5
        assert s.metadata == {"key": "val"}

    def test_score_defaults(self):
        s = Score(name="test", value=1.0)
        assert s.metadata == {}


class TestExactMatch:
    def test_match(self):
        scorer = ExactMatch()
        result = scorer(input="x", output="hello", expected="hello")
        assert result.value == 1.0
        assert result.name == "ExactMatch"

    def test_no_match(self):
        scorer = ExactMatch()
        result = scorer(input="x", output="hello", expected="world")
        assert result.value == 0.0

    def test_coerces_to_string(self):
        scorer = ExactMatch()
        result = scorer(input="x", output=42, expected=42)
        assert result.value == 1.0

    def test_none_output(self):
        scorer = ExactMatch()
        result = scorer(input="x", output=None, expected="hello")
        assert result.value == 0.0

    def test_satisfies_protocol(self):
        assert isinstance(ExactMatch(), Scorer)


class TestContains:
    def test_contains(self):
        scorer = Contains()
        result = scorer(input="x", output="Hello, World!", expected="world")
        assert result.value == 1.0  # case-insensitive

    def test_not_contains(self):
        scorer = Contains()
        result = scorer(input="x", output="Hello", expected="world")
        assert result.value == 0.0

    def test_case_sensitive(self):
        scorer = Contains(case_sensitive=True)
        result = scorer(input="x", output="Hello, World!", expected="world")
        assert result.value == 0.0  # "world" != "World"

    def test_case_sensitive_match(self):
        scorer = Contains(case_sensitive=True)
        result = scorer(input="x", output="Hello, World!", expected="World")
        assert result.value == 1.0

    def test_none_output(self):
        scorer = Contains()
        result = scorer(input="x", output=None, expected="hello")
        assert result.value == 0.0

    def test_satisfies_protocol(self):
        assert isinstance(Contains(), Scorer)


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_clean_json(self):
        score, reasoning = _parse_judge_response('{"score": 0.8, "reasoning": "Good answer"}')
        assert score == 0.8
        assert reasoning == "Good answer"

    def test_json_in_code_block(self):
        text = '```json\n{"score": 0.7, "reasoning": "Mostly correct"}\n```'
        score, reasoning = _parse_judge_response(text)
        assert score == 0.7
        assert reasoning == "Mostly correct"

    def test_json_in_plain_code_block(self):
        text = '```\n{"score": 0.5, "reasoning": "Partial"}\n```'
        score, reasoning = _parse_judge_response(text)
        assert score == 0.5

    def test_score_regex_fallback(self):
        text = 'The score is {"score": 0.9, "reasoning": "great"} end'
        score, reasoning = _parse_judge_response(text)
        assert score == 0.9

    def test_unparseable_returns_zero(self):
        score, reasoning = _parse_judge_response("I think it's pretty good")
        assert score == 0.0
        assert "Parse error" in reasoning

    def test_integer_score(self):
        score, _ = _parse_judge_response('{"score": 1, "reasoning": "Perfect"}')
        assert score == 1.0


# ---------------------------------------------------------------------------
# LlmJudge
# ---------------------------------------------------------------------------


class TestLlmJudge:
    def _make_judge(self, response_text: str, **kwargs) -> LlmJudge:
        """Create an LlmJudge with _call_llm mocked."""
        judge = LlmJudge(**kwargs)
        judge._call_llm = MagicMock(return_value=response_text)
        return judge

    def test_basic(self):
        judge = self._make_judge(
            '{"score": 0.9, "reasoning": "Correct answer"}',
            model="openai/gpt-4o-mini",
        )
        result = judge(input="What is 2+2?", output="4", expected="4")

        assert result.value == 0.9
        assert result.name == "LlmJudge"
        assert result.metadata["reasoning"] == "Correct answer"
        judge._call_llm.assert_called_once()

    def test_custom_name(self):
        judge = self._make_judge(
            '{"score": 1.0, "reasoning": "Perfect"}',
            model="openai/gpt-4o-mini",
            name="Factuality",
        )
        result = judge(input="x", output="y", expected="y")
        assert result.name == "Factuality"

    def test_builtin_prompt_key(self):
        judge = self._make_judge(
            '{"score": 0.5, "reasoning": "Partial"}',
            model="test",
            prompt_template="closedqa",
        )
        judge(input="Q", output="A", expected="A")

        prompt = judge._call_llm.call_args[0][0]
        assert "correctly answered" in prompt

    def test_custom_prompt_template(self):
        template = "Rate: {input} -> {output} (expected: {expected})\nJSON only: "
        judge = self._make_judge(
            '{"score": 0.8, "reasoning": "Good"}',
            model="test",
            prompt_template=template,
        )
        judge(input="Q", output="A", expected="A")

        prompt = judge._call_llm.call_args[0][0]
        assert "Rate: Q -> A (expected: A)" in prompt

    def test_llm_error_returns_zero(self):
        judge = LlmJudge(model="test")
        judge._call_llm = MagicMock(side_effect=Exception("API error"))

        result = judge(input="x", output="y", expected="y")
        assert result.value == 0.0
        assert "error" in result.metadata

    def test_score_clamped(self):
        judge = self._make_judge('{"score": 1.5, "reasoning": "Over"}', model="test")
        result = judge(input="x", output="y", expected="y")
        assert result.value == 1.0

    def test_score_clamped_negative(self):
        judge = self._make_judge('{"score": -0.5, "reasoning": "Under"}', model="test")
        result = judge(input="x", output="y", expected="y")
        assert result.value == 0.0

    def test_none_expected(self):
        judge = self._make_judge('{"score": 0.6, "reasoning": "Decent"}', model="test")
        judge(input="x", output="y", expected=None)

        prompt = judge._call_llm.call_args[0][0]
        assert "(none)" in prompt

    def test_temperature_stored(self):
        judge = LlmJudge(model="test", temperature=0.7)
        assert judge.temperature == 0.7

    def test_judge_prompts_keys(self):
        assert "factuality" in JUDGE_PROMPTS
        assert "closedqa" in JUDGE_PROMPTS
        assert "quality" in JUDGE_PROMPTS
