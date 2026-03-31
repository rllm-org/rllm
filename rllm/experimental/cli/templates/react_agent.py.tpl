"""{{project_name}}: A multi-turn ReAct agent for rllm eval.

Usage:
    rllm model setup
    rllm eval hotpotqa --agent {{module_name}}.agent:{{agent_instance}}
"""

import json

from openai import OpenAI

from rllm.experimental.eval.types import AgentConfig
from rllm.types import Episode, Step, Trajectory


SYSTEM_PROMPT = """\
You are a ReAct agent. You solve problems by alternating between Thought, Action, and Observation steps.

Available tools:
- search(query: str) -> str: Search for information

Respond in this format:
Thought: <your reasoning>
Action: {"tool": "search", "args": {"query": "..."}}

When you have the final answer:
Thought: I now know the answer.
Answer: <your answer>"""

MAX_TURNS = 5


def _parse_action(text: str) -> dict | None:
    """Extract the JSON action from the model response."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            try:
                return json.loads(line[len("Action:"):].strip())
            except json.JSONDecodeError:
                return None
    return None


def _execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool and return the observation. Replace with your own tools."""
    if tool_name == "search":
        # TODO: Replace with your actual search implementation
        return f"[Search result for: {args.get('query', '')}]"
    return f"Unknown tool: {tool_name}"


class {{agent_class}}:
    """Multi-turn ReAct agent with tool use."""

    def run(self, task: dict, config: AgentConfig) -> Episode:
        client = OpenAI(base_url=config.base_url, api_key="EMPTY")
        question = task.get("question", "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: list[Step] = []
        final_answer = ""

        for turn in range(MAX_TURNS):
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
            )
            output = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": output})

            # Check for final answer
            if "Answer:" in output:
                final_answer = output.split("Answer:")[-1].strip()
                steps.append(Step(input=question if turn == 0 else "", output=output, done=True))
                break

            # Parse and execute action
            action = _parse_action(output)
            if action:
                observation = _execute_tool(action["tool"], action.get("args", {}))
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                steps.append(Step(input=question if turn == 0 else "", output=output, done=False))
            else:
                # No action found, treat as final answer
                final_answer = output
                steps.append(Step(input=question if turn == 0 else "", output=output, done=True))
                break
        else:
            # Max turns reached
            steps.append(Step(input="", output=output, done=True))

        traj = Trajectory(name="solver", steps=steps)
        return Episode(task=task, trajectories=[traj], artifacts={"answer": final_answer})


{{agent_instance}} = {{agent_class}}()
