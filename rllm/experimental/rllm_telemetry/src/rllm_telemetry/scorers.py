"""Scorer protocol and built-in scorers for experiment evaluation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("rllm_telemetry.scorers")


@dataclass
class Score:
    """Result of a single scorer applied to one test case.

    Attributes:
        name: Scorer name (e.g. ``"ExactMatch"``).
        value: Numeric score, typically 0.0 to 1.0.
        metadata: Optional extra information from the scorer.
    """

    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    """Protocol that all scorers must satisfy.

    A scorer is any callable with this signature::

        scorer(*, input: Any, output: Any, expected: Any) -> Score

    The ``name`` attribute is used for display and storage.
    """

    name: str

    def __call__(self, *, input: Any, output: Any, expected: Any) -> Score: ...


class ExactMatch:
    """Scores 1.0 if ``str(output) == str(expected)``, else 0.0."""

    name: str = "ExactMatch"

    def __call__(self, *, input: Any, output: Any, expected: Any) -> Score:
        match = str(output) == str(expected)
        return Score(name=self.name, value=1.0 if match else 0.0)


class Contains:
    """Scores 1.0 if ``str(expected)`` is a substring of ``str(output)``, else 0.0.

    Case-insensitive by default.
    """

    name: str = "Contains"

    def __init__(self, *, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def __call__(self, *, input: Any, output: Any, expected: Any) -> Score:
        out = str(output)
        exp = str(expected)
        if not self.case_sensitive:
            out = out.lower()
            exp = exp.lower()
        match = exp in out
        return Score(name=self.name, value=1.0 if match else 0.0)


# ---------------------------------------------------------------------------
# Default judge prompts
# ---------------------------------------------------------------------------

_FACTUALITY_PROMPT = """\
You are an expert judge evaluating the factual accuracy of an AI assistant's response.

**Input:** {input}
**Expected answer:** {expected}
**Actual output:** {output}

Rate the factual accuracy of the actual output compared to the expected answer.
Consider:
- Does the output contain the correct information from the expected answer?
- Is the output free of factual errors or contradictions?
- Partial credit is appropriate if some facts are correct but others are missing or wrong.

Respond with ONLY a JSON object (no markdown, no explanation):
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>"}}
"""

_CLOSEDQA_PROMPT = """\
You are an expert judge evaluating whether an AI assistant correctly answered a question.

**Question:** {input}
**Reference answer:** {expected}
**Assistant's answer:** {output}

Does the assistant's answer correctly address the question based on the reference answer?
- 1.0: Fully correct and complete
- 0.5-0.9: Partially correct or missing minor details
- 0.1-0.4: Mostly incorrect but has some relevant content
- 0.0: Completely wrong or irrelevant

Respond with ONLY a JSON object (no markdown, no explanation):
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>"}}
"""

_QUALITY_PROMPT = """\
You are an expert judge evaluating the overall quality of an AI assistant's response.

**Input:** {input}
**Output:** {output}

Rate the response on overall quality considering:
- Helpfulness and relevance to the input
- Clarity and coherence
- Completeness

Respond with ONLY a JSON object (no markdown, no explanation):
{{"score": <float 0.0 to 1.0>, "reasoning": "<brief explanation>"}}
"""

JUDGE_PROMPTS: dict[str, str] = {
    "factuality": _FACTUALITY_PROMPT,
    "closedqa": _CLOSEDQA_PROMPT,
    "quality": _QUALITY_PROMPT,
}


# ---------------------------------------------------------------------------
# LlmJudge scorer
# ---------------------------------------------------------------------------


def _parse_judge_response(text: str) -> tuple[float, str]:
    """Extract score and reasoning from the judge LLM response."""
    # Try direct JSON parse first
    try:
        data = json.loads(text)
        return float(data["score"]), data.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try extracting JSON from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return float(data["score"]), data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Try finding any JSON object in the text
    m = re.search(r'\{\s*"score"\s*:\s*([\d.]+).*?\}', text, re.DOTALL)
    if m:
        try:
            return float(m.group(1)), ""
        except ValueError:
            pass

    logger.warning("Could not parse judge response: %s", text[:200])
    return 0.0, f"Parse error: {text[:200]}"


class LlmJudge:
    """Scorer that uses an LLM to evaluate output quality.

    Uses ``litellm.completion`` so any model supported by LiteLLM works
    (OpenAI, Anthropic, Google, local models, etc.).

    Args:
        model: Model identifier (e.g. ``"openai/gpt-4o-mini"``,
            ``"anthropic/claude-haiku-4-5-20251001"``).
        prompt_template: Prompt template with ``{input}``, ``{output}``, and
            ``{expected}`` placeholders. Defaults to the ``"factuality"``
            built-in prompt. Can also pass a key from :data:`JUDGE_PROMPTS`.
        name: Scorer name for display. Defaults to ``"LlmJudge"``.
        temperature: Sampling temperature for the judge model.

    Example::

        judge = LlmJudge(model="openai/gpt-4o-mini")
        result = Eval(
            "My Eval",
            data=[...],
            task=my_agent,
            scores=[judge],
        )

    Using a built-in prompt::

        judge = LlmJudge(model="openai/gpt-4o-mini", prompt_template="closedqa")
    """

    name: str

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        *,
        prompt_template: str | None = None,
        name: str = "LlmJudge",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.name = name
        self.temperature = temperature

        if prompt_template is None:
            self._template = _FACTUALITY_PROMPT
        elif prompt_template in JUDGE_PROMPTS:
            self._template = JUDGE_PROMPTS[prompt_template]
        else:
            self._template = prompt_template

    def _call_llm(self, prompt: str) -> str:
        """Call the judge LLM and return the response text.

        Separated for testability — mock this method in tests.
        """
        try:
            import litellm
        except ImportError as err:
            raise ImportError("LlmJudge requires litellm. Install it with: pip install litellm") from err

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def __call__(self, *, input: Any, output: Any, expected: Any) -> Score:
        prompt = self._template.format(
            input=input,
            output=output,
            expected=expected if expected is not None else "(none)",
        )

        try:
            text = self._call_llm(prompt)
        except ImportError:
            raise
        except Exception as exc:
            logger.warning("LlmJudge call failed: %s", exc)
            return Score(
                name=self.name,
                value=0.0,
                metadata={"error": str(exc)},
            )

        score_val, reasoning = _parse_judge_response(text)
        # Clamp to [0, 1]
        score_val = max(0.0, min(1.0, score_val))

        return Score(
            name=self.name,
            value=score_val,
            metadata={"reasoning": reasoning, "raw_response": text},
        )
