"""GAIA deep-research agent for `rllm eval gaia`.

A minimal multi-turn ReAct tool-calling agent matching how frontier models are
evaluated on GAIA: a web **search** + page **browse** loop (cf. HuggingFace
Open Deep Research and Princeton HAL, the two reference GAIA scaffolds). Tools
are rLLM's Tavily search + extract — a single `TAVILY_API_KEY` covers both.

Code-execution / file-parsing / multimodal tools are intentionally out of scope
here: this targets the text-only GAIA subset produced by `gaia_transform`
(file-attachment tasks are skipped). Those tools are the natural follow-up for
the harder GAIA levels.

Run:
    export TAVILY_API_KEY=...      # web search + page extract
    export HF_TOKEN=...            # GAIA dataset is gated
    rllm model setup               # configure your model provider
    rllm eval gaia --agent cookbooks.gaia.agent:agent --max-examples 5
"""

from __future__ import annotations

import json

from rllm.eval.reward_fns.gaia import SYSTEM_PROMPT
from rllm.tools.web_tools.tavily_tool import TavilyExtractTool, TavilySearchTool
from rllm.types import Episode, Step, Trajectory

MAX_TURNS = 12
_MAX_OBS_CHARS = 6000  # cap tool observations so browse output can't blow up context


def _tool_observation(tool, args: dict) -> str:
    """Run a tool and return a string observation (truncated)."""
    out = tool.forward(**args)
    payload = out.error if out.error else out.output
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return text[:_MAX_OBS_CHARS]


def run_tool_loop(client, model: str, tools: list, question: str, *, system_prompt: str = SYSTEM_PROMPT, max_turns: int = MAX_TURNS) -> tuple[list[Step], str]:
    """Multi-turn OpenAI tool-calling loop. Returns (steps, final_answer).

    Factored out from `GaiaAgent.run` so the control flow is unit-testable with a
    fake client + fake tools (no network / no API keys).
    """
    schemas = [t.json for t in tools]
    tool_map = {t.json["function"]["name"]: t for t in tools}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(question)},
    ]
    steps: list[Step] = []
    answer = ""

    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, tools=schemas, temperature=0.0)
        except Exception as exc:  # surface model errors as a terminal step, don't crash the run
            steps.append(Step(input=f"turn_{turn}", output=f"LLM error: {exc}", done=True))
            break

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:  # no tool call -> final answer
            answer = msg.content or ""
            steps.append(Step(input=f"turn_{turn}", output=answer, done=True))
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool = tool_map.get(name)
            obs = f"Unknown tool: {name}" if tool is None else _tool_observation(tool, args)
            steps.append(Step(input=tc.function.arguments, output=obs))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})

    return steps, answer


class GaiaAgent:
    """Multi-turn search + browse ReAct agent (Tavily tools)."""

    name = "gaia"
    max_concurrent = 8

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns

    def run(self, task, config) -> Episode:
        from openai import OpenAI

        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        tools = [TavilySearchTool(), TavilyExtractTool()]

        question = getattr(task, "instruction", None)
        if question is None and isinstance(task, dict):
            question = task.get("question", "")

        steps, answer = run_tool_loop(client, config.model, tools, str(question or ""), max_turns=self.max_turns)

        uid = getattr(config, "session_uid", "")
        task_id = getattr(task, "id", "") if not isinstance(task, dict) else task.get("task_id", "")
        traj = Trajectory(uid=uid, name=self.name, task=task_id, steps=steps, output=answer)
        return Episode(id=uid, task=task_id, trajectories=[traj], artifacts={"answer": answer})


agent = GaiaAgent()
