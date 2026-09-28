from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).parent
FINQA = Path("/home/yanan/agents/rllm/cookbooks/finqa")
for path in (ROOT, FINQA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import multitable_v2_flow as flow  # noqa: E402
from rllm.types import AgentConfig, Task  # noqa: E402


def _tool_message(index: int):
    call = SimpleNamespace(
        id=f"call-{index}",
        type="function",
        function=SimpleNamespace(name="calculator", arguments=json.dumps({"expression": "1+1"})),
    )
    return SimpleNamespace(role="assistant", content="", tool_calls=[call])


def _final_message():
    return SimpleNamespace(role="assistant", content="FINAL ANSWER: complete", tool_calls=[])


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        msg = _tool_message(len(self.calls)) if "tools" in kwargs else _final_message()
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeClient:
    last = None

    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        type(self).last = self


class _MalformedThenValidCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if len(self.calls) == 1:
            msg = SimpleNamespace(
                role="assistant",
                content='<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}</tool_call>',
                tool_calls=[],
            )
        elif len(self.calls) == 2:
            msg = _tool_message(2)
        else:
            msg = _final_message()
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _MalformedThenValidClient:
    last = None

    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(completions=_MalformedThenValidCompletions())
        type(self).last = self


class MultiTableV2Tests(unittest.TestCase):
    def test_protocol_constants_and_prompt_contract(self):
        self.assertEqual(flow.MAX_TURNS, 50)
        self.assertEqual(flow.MAX_TOOL_TURNS, 45)
        self.assertEqual(flow.FINAL_ATTEMPTS, 5)
        self.assertEqual(flow.FINAL_MAX_COMPLETION_TOKENS, 8192)
        self.assertIn("strict answer template", flow.SYSTEM_PROMPT)
        self.assertIn("internal checklist", flow.SYSTEM_PROMPT)
        self.assertIn("FINAL ANSWER:", flow.SYSTEM_PROMPT)
        self.assertIn("reference solutions", flow.SYSTEM_PROMPT)

    def test_non_object_tool_arguments_fail_closed(self):
        for raw, typename in [('"table"', "str"), ("[]", "list"), ("null", "NoneType")]:
            with self.subTest(raw=raw):
                tc = SimpleNamespace(function=SimpleNamespace(name="get_table_info", arguments=raw))
                output, failed = flow._exec_tool_call(tc, [])
                self.assertTrue(failed)
                self.assertIn(f"got {typename}", output)

    def test_unparsed_literal_tool_call_is_retried_not_accepted_as_final(self):
        task = Task(
            id="multi-malformed",
            instruction="unused",
            metadata={"question": "Fill the template.", "question_type": "multi_table", "company": "acme"},
        )
        with patch.object(flow, "AsyncOpenAI", _MalformedThenValidClient):
            episode = asyncio.run(
                flow.finqa_multitable_v2._fn(
                    task, AgentConfig(model="fake", base_url="http://fake", session_uid="test-session")
                )
            )
        self.assertEqual(episode.artifacts["answer"], "FINAL ANSWER: complete")
        self.assertEqual(episode.artifacts["malformed_tool_calls"], 1)
        self.assertEqual(episode.artifacts["tool_calls"], 1)
        calls = _MalformedThenValidClient.last.chat.completions.calls
        self.assertEqual(len(calls), 3)
        self.assertIn("could not be executed", calls[1]["messages"][-1]["content"])

    def test_45_tool_turns_then_forced_final_stays_within_50(self):
        task = Task(
            id="multi-1",
            instruction="unused",
            metadata={"question": "Fill the template.", "question_type": "multi_table_hard", "company": "acme"},
        )
        with (
            patch.object(flow, "AsyncOpenAI", _FakeClient),
            patch.object(flow, "FINALIZE_AT_TRANSCRIPT_CHARS", 10_000_000),
        ):
            episode = asyncio.run(
                flow.finqa_multitable_v2._fn(
                    task, AgentConfig(model="fake", base_url="http://fake", session_uid="test-session")
                )
            )
        self.assertEqual(episode.artifacts["turns"], 46)
        self.assertEqual(episode.artifacts["tool_calls"], 45)
        self.assertEqual(episode.artifacts["max_turns"], 50)
        self.assertEqual(episode.artifacts["answer"], "FINAL ANSWER: complete")
        self.assertFalse(episode.artifacts["final_fallback_used"])
        calls = _FakeClient.last.chat.completions.calls
        self.assertEqual(len(calls), 46)
        self.assertIn("tools", calls[44])
        self.assertNotIn("tools", calls[45])
        self.assertEqual(calls[45]["max_completion_tokens"], 8192)
        self.assertIn("Canonical tool company identifier", calls[0]["messages"][1]["content"])
        self.assertIn("`acme`", calls[0]["messages"][1]["content"])

    def test_rejects_non_multitable_task(self):
        task = Task(
            id="single",
            instruction="q",
            metadata={"question": "q", "question_type": "single_table", "company": "acme"},
        )
        with patch.object(flow, "AsyncOpenAI", _FakeClient):
            with self.assertRaisesRegex(ValueError, "only accepts multi_table"):
                asyncio.run(
                    flow.finqa_multitable_v2._fn(
                        task, AgentConfig(model="fake", base_url="http://fake", session_uid="test-session")
                    )
                )

    def test_missing_canonical_company_fails_closed(self):
        task = Task(
            id="missing-company",
            instruction="q",
            metadata={"question": "q", "question_type": "multi_table_hard"},
        )
        with patch.object(flow, "AsyncOpenAI", _FakeClient):
            with self.assertRaisesRegex(ValueError, "canonical company identifier"):
                asyncio.run(
                    flow.finqa_multitable_v2._fn(
                        task, AgentConfig(model="fake", base_url="http://fake", session_uid="test-session")
                    )
                )


if __name__ == "__main__":
    unittest.main()
