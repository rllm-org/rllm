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
import logging

from rllm.tools.web_tools.tavily_tool import TavilyExtractTool, TavilySearchTool
from rllm.types import Episode, Step, Trajectory

logger = logging.getLogger(__name__)

MAX_TURNS = 12
_MAX_OBS_CHARS = 6000  # cap tool observations so browse output can't blow up context

# The agent prompt MUST tell the model it has tools and to use them — GAIA questions
# need fresh facts, and most models won't call tools unless explicitly instructed.
AGENT_SYSTEM_PROMPT = (
    "You are a general AI assistant answering questions that require accurate, "
    "up-to-date facts from the web. You have two tools:\n"
    "- tavily-search(query): search the web for relevant sources.\n"
    "- tavily-extract(urls): read the full text of specific web pages.\n"
    "Always use these tools to look things up before answering — do NOT answer from "
    "memory. Search first, then extract the most relevant pages, then reason.\n\n"
    "When you are confident, give your answer on its own line as:\n"
    "FINAL ANSWER: <answer>\n"
    "The answer should be a number, as few words as possible, or a comma-separated "
    "list of numbers and/or strings. Do not add units unless asked."
)


def _tool_observation(tool, args: dict) -> str:
    """Run a tool and return a string observation (truncated). Never raises: a tool
    failure (missing API key, bad model-emitted kwargs, network error) becomes an
    error observation the model can react to, not a crashed rollout — matching
    ToolCallingMixin's behavior (rllm/harnesses/tool_calling.py)."""
    try:
        out = tool.forward(**args)
    except Exception as exc:
        return f"Tool error: {type(exc).__name__}: {exc}"[:_MAX_OBS_CHARS]
    payload = out.error if out.error else out.output
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return text[:_MAX_OBS_CHARS]


def run_tool_loop(client, model: str, tools: list, question: str, *, system_prompt: str = AGENT_SYSTEM_PROMPT, max_turns: int = MAX_TURNS) -> tuple[list[Step], str]:
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
        except Exception as exc:
            # Don't record a Step for a failed call: no gateway trace exists for it, and
            # the enricher requires step<->trace parity (rllm/engine/agentflow_engine.py).
            logger.warning("LLM call failed on turn %d: %s", turn, exc)
            break

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:  # no tool call -> final answer
            answer = msg.content or ""
            steps.append(Step(input=f"turn_{turn}", output=answer, done=True))
            break

        # Execute every tool call in this turn, but record exactly ONE Step per turn:
        # the eval trace-enricher consumes traces 1:1 with agent steps (one LLM call ==
        # one Step), so a step-per-tool-call inflates the count on parallel tool calls
        # and raises EnrichMismatchError. See rllm/engine/agentflow_engine.py.
        observations = []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tool = tool_map.get(name)
            obs = f"Unknown tool: {name}" if tool is None else _tool_observation(tool, args)
            observations.append(f"{name}: {obs}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
        steps.append(Step(input=f"turn_{turn}", output="\n".join(observations)))

    # max_turns exhausted without a final answer: mark the episode terminated on the
    # LAST recorded step (flipping the flag keeps step<->trace parity; appending a new
    # step would break it). `answer` stays "" and is scored as an empty answer via
    # Episode.artifacts["answer"].
    if steps and not steps[-1].done:
        steps[-1].done = True

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
