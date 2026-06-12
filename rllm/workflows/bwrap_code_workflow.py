"""BwrapCodeWorkflow: single-turn code execution in bwrap sandbox.

LLM generates bash/python code -> sandbox executes -> output compared to expected.
Designed as a lightweight landmark for local sandbox RL training on Koala.
"""

from __future__ import annotations

import re
import shlex

from rllm.agents.agent import BaseAgent
from rllm.engine import ModelOutput, RolloutEngine
from rllm.sandbox.backends.bwrap import BwrapSandbox, is_available
from rllm.types import Action, Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

_SYSTEM_PROMPT = """You are a coding assistant. Solve the task by writing code.
Wrap your solution in a ```bash or ```python code block. Only output the code block."""


class _SimpleAgent(BaseAgent):
    def __init__(self, **kwargs):
        self._trajectory = Trajectory()

    def reset(self):
        self._trajectory = Trajectory()

    def update_from_model(self, *args, **kwargs):
        pass

    def update_from_env(self, *args, **kwargs):
        pass

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory


class BwrapCodeWorkflow(Workflow):
    def __init__(self, exec_timeout: float = 30, **kwargs):
        super().__init__(**kwargs)
        self.exec_timeout = exec_timeout
        self.agent = _SimpleAgent()
        if not is_available():
            raise RuntimeError("bwrap not installed. Run: apt-get install -y bubblewrap")

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        prompt = task.get("prompt") or task.get("question") or task.get("problem")
        if not prompt:
            raise ValueError("Task must have 'prompt', 'question', or 'problem' key")
        expected = task.get("expected_output", "").strip()

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        output: ModelOutput = await self.rollout_engine.get_model_response(
            messages, application_id=uid, **kwargs
        )
        lang, code = _extract_code(output.content)

        sandbox = BwrapSandbox(name=f"wf-{uid[:8]}")
        try:
            if code:
                if lang in ("bash", "sh"):
                    exec_cmd = f"bash -c {shlex.quote(code)}"
                else:
                    exec_cmd = f"python3 -c {shlex.quote(code)}"
                try:
                    stdout = sandbox.exec(exec_cmd, timeout=self.exec_timeout)
                except Exception as e:
                    stdout = f"ERROR: {e}"
            else:
                stdout = "ERROR: no code block found"
        finally:
            sandbox.close()

        actual = stdout.strip()
        reward = 1.0 if actual == expected else 0.0

        trajectory = self.agent.trajectory
        trajectory.steps.append(
            Step(
                chat_completions=messages + [{"role": "assistant", "content": output.content}],
                action=Action(action=output.content),
                reward=reward,
                model_output=output,
            )
        )

        self.commit(agent=self.agent, reset=True)
        raise TerminationEvent(TerminationReason.ENV_DONE)


def _extract_code(text: str) -> tuple[str, str | None]:
    """Extract (language, code) from a fenced code block."""
    match = re.search(r"```(python3?|bash|sh)?[ \t]*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        lang = (match.group(1) or "python").lower()
        if lang == "python3":
            lang = "python"
        return lang, match.group(2).strip()
    return "python", None
