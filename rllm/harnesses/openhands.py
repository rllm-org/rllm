"""Run the OpenHands Software Agent SDK inside an rLLM sandbox.

The agent and repository share the sandbox filesystem. OpenHands talks to the
rLLM gateway through its OpenAI-compatible endpoint, so the engine captures the
LLM trajectory while the task's normal verifier remains authoritative.
"""

from __future__ import annotations

import json
import logging
import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Task

logger = logging.getLogger(__name__)

# Pin the SDK and its tmux parser so snapshots remain reproducible. The SDK's
# tools package allows newer libtmux versions that do not parse a few benchmark
# images correctly, hence the explicit transitive pin.
OPENHANDS_SDK_REVISION = "43376f1868ffd702746080714a59c16d3f69ec12"
OPENHANDS_EXTENSIONS_REVISION = "2cfd75e2396ad5111f0e1670a5534a79bbf97a3e"
OPENHANDS_LIBTMUX_VERSION = "0.53.0"
OPENHANDS_LOCALE = "C.UTF-8"

_VENV_DIR = "/opt/rllm/openhands"
_DRIVER_PATH = "/tmp/rllm-openhands/driver.py"
_PROMPT_PATH = "/tmp/rllm-openhands/prompt.txt"
_SUMMARY_PATH = "/tmp/rllm-openhands/summary.json"
_OUTCOME_PATH = "/tmp/rllm-openhands/outcome.json"
_REVISION_FILE = f"{_VENV_DIR}/.sdk-revision"
_INSTALL_REVISION = f"{OPENHANDS_SDK_REVISION}:libtmux=={OPENHANDS_LIBTMUX_VERSION}"
_SDK_ARCHIVE = f"https://github.com/OpenHands/software-agent-sdk/archive/{OPENHANDS_SDK_REVISION}.tar.gz"
_SDK_REQUIREMENT = f"openhands-sdk @ {_SDK_ARCHIVE}#subdirectory=openhands-sdk"
_TOOLS_REQUIREMENT = f"openhands-tools @ {_SDK_ARCHIVE}#subdirectory=openhands-tools"

_INSTALL_SCRIPT = rf"""
set -e
export DEBIAN_FRONTEND=noninteractive

if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq curl ca-certificates git tmux
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl ca-certificates git tmux bash
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    for attempt in 1 2 3 4 5; do
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            break
        fi
        if [ "$attempt" -eq 5 ]; then
            echo "uv install failed after 5 attempts" >&2
            exit 1
        fi
        sleep $((attempt * 3))
    done
fi
export PATH="$HOME/.local/bin:$PATH"

installed_revision="$(cat {_REVISION_FILE} 2>/dev/null || true)"
if [ "$installed_revision" != "{_INSTALL_REVISION}" ]; then
    uv venv --python 3.12 --clear {_VENV_DIR}
    uv pip install --python {_VENV_DIR}/bin/python {shlex.quote(_SDK_REQUIREMENT)}
    uv pip install --python {_VENV_DIR}/bin/python --no-deps {shlex.quote(_TOOLS_REQUIREMENT)}
    uv pip install --python {_VENV_DIR}/bin/python binaryornot cachetools libtmux=={OPENHANDS_LIBTMUX_VERSION} func-timeout
    printf '%s\n' "{_INSTALL_REVISION}" > {_REVISION_FILE}
fi
"""

_DRIVER_SCRIPT = r'''"""Local-workspace OpenHands driver for rLLM."""

import json
import os
import traceback

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.context import AgentContext
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.skills import load_public_skills
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.tools.preset.default import get_default_tools


def _finished(events):
    for event in reversed(events):
        if isinstance(event, ActionEvent):
            return isinstance(event.action, FinishAction)
    return False


def _sent_message(events):
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.source == "agent":
            return True
        if isinstance(event, ActionEvent):
            return False
    return False


def _fake_response(conversation):
    user_messages = [
        event
        for event in conversation.state.events
        if isinstance(event, MessageEvent) and event.source == "user"
    ]
    message = (
        "Please continue working on the task on whatever approach you think is suitable.\n"
        "When you think you have solved the question, please use the finish tool and "
        "include your final answer in the message parameter of the finish tool.\n"
        "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n"
    )
    if len(user_messages) >= 2:
        message += 'If you want to give up, use the "finish" tool to finish the interaction.\n'
    return message


class ConversationIncompleteError(RuntimeError):
    pass


def _write_json(path, data):
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _optional_int(name):
    value = os.environ.get(name)
    return int(value) if value else None


def main():
    conversation = None
    failure = None
    failure_traceback = None
    fake_responses = 0
    try:
        llm_kwargs = {
            "usage_id": "agent",
            "model": os.environ["LLM_MODEL"],
            "api_key": os.environ["LLM_API_KEY"],
            "base_url": os.environ["LLM_BASE_URL"],
            "temperature": float(os.environ["RLLM_OPENHANDS_TEMPERATURE"]),
            "top_p": float(os.environ["RLLM_OPENHANDS_TOP_P"]),
            "disable_vision": True,
            "litellm_extra_body": json.loads(
                os.environ.get("RLLM_OPENHANDS_LITELLM_EXTRA_BODY", "{}")
            ),
        }
        max_input_tokens = _optional_int("RLLM_OPENHANDS_MAX_INPUT_TOKENS")
        max_output_tokens = _optional_int("RLLM_OPENHANDS_MAX_OUTPUT_TOKENS")
        if max_input_tokens is not None:
            llm_kwargs["max_input_tokens"] = max_input_tokens
        if max_output_tokens is not None:
            llm_kwargs["max_output_tokens"] = max_output_tokens

        llm = LLM(**llm_kwargs)
        condenser_llm = llm.model_copy(deep=True, update={"usage_id": "condenser"})
        public_skills = load_public_skills()
        agent_context = AgentContext(skills=public_skills) if public_skills else None
        agent = Agent(
            llm=llm,
            tools=get_default_tools(enable_browser=False),
            system_prompt_kwargs={"cli_mode": True},
            condenser=LLMSummarizingCondenser(
                llm=condenser_llm,
                max_size=int(os.environ["RLLM_OPENHANDS_CONDENSER_MAX_SIZE"]),
                keep_first=int(os.environ["RLLM_OPENHANDS_CONDENSER_KEEP_FIRST"]),
            ),
            agent_context=agent_context,
        )
        conversation = Conversation(
            agent=agent,
            workspace=os.environ["RLLM_OPENHANDS_WORKDIR"],
            max_iteration_per_run=int(os.environ["RLLM_OPENHANDS_MAX_ITERATIONS"]),
            delete_on_close=True,
        )
        with open(os.environ["RLLM_OPENHANDS_PROMPT_PATH"], encoding="utf-8") as stream:
            instruction = stream.read()
        conversation.send_message(instruction)

        while True:
            conversation.run()
            events = list(conversation.state.events)
            if conversation.state.execution_status != ConversationExecutionStatus.FINISHED:
                break
            if _finished(events) or not _sent_message(events) or fake_responses >= 10:
                break
            conversation.send_message(_fake_response(conversation))
            fake_responses += 1
    except BaseException as exc:
        # OpenHands may raise while processing a terminal FinishAction. The
        # typed event remains authoritative, while pre-finish failures remain
        # visible in the structured outcome and full traceback.
        failure = exc
        failure_traceback = traceback.format_exc()

    events = list(conversation.state.events) if conversation is not None else []
    finished = _finished(events)
    execution_status = None
    if conversation is not None:
        status = conversation.state.execution_status
        execution_status = getattr(status, "value", str(status))
    if not finished and failure is None:
        failure = ConversationIncompleteError(
            f"OpenHands stopped without FinishAction (execution_status={execution_status})"
        )

    summary = {
        "execution_status": execution_status,
        "events": len(events),
        "fake_responses": fake_responses,
        "finished": finished,
    }
    try:
        _write_json(os.environ["RLLM_OPENHANDS_SUMMARY_PATH"], summary)
    except BaseException as exc:
        if failure is None:
            failure = exc
            failure_traceback = traceback.format_exc()

    outcome = {
        "finished": finished,
        "execution_status": execution_status,
        "exception_type": None if finished else type(failure).__name__,
        "message": "" if failure is None else str(failure),
        "traceback": None if finished else failure_traceback,
    }
    _write_json(os.environ["RLLM_OPENHANDS_OUTCOME_PATH"], outcome)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
'''


class OpenHandsHarness(BaseCliHarness):
    """Run OpenHands against an arbitrary sandbox task and gateway model."""

    name = "openhands"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/openhands.log"

    temperature: float = 1.0
    top_p: float = 1.0
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_iterations: int = 500
    condenser_max_size: int = 240
    condenser_keep_first: int = 2
    litellm_extra_body: dict | None = None

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def render_prompt(self, task: Task, workdir: str) -> str:
        """Return the prompt written for OpenHands; profiles may override it."""
        del workdir
        return str(task.instruction).strip()

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        api_key = self.gateway_api_key(config, "RLLM_GATEWAY_API_KEY")
        workdir = str(task.metadata.get("workdir") or "/app")
        env = {
            # Always use LiteLLM's OpenAI adapter for the sandbox-to-gateway
            # hop. Preserve config.model verbatim after that single prefix so
            # the gateway sees exactly the alias it registered.
            "LLM_MODEL": f"openai/{config.model}",
            "LLM_API_KEY": api_key,
            "LLM_BASE_URL": config.base_url,
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_BASE": config.base_url,
            "OPENAI_BASE_URL": config.base_url,
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "EXTENSIONS_REF": OPENHANDS_EXTENSIONS_REVISION,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LC_ALL": OPENHANDS_LOCALE,
            "LANG": OPENHANDS_LOCALE,
            "RLLM_OPENHANDS_WORKDIR": workdir,
            "RLLM_OPENHANDS_PROMPT_PATH": _PROMPT_PATH,
            "RLLM_OPENHANDS_SUMMARY_PATH": _SUMMARY_PATH,
            "RLLM_OPENHANDS_OUTCOME_PATH": _OUTCOME_PATH,
            "RLLM_OPENHANDS_TEMPERATURE": str(self.temperature),
            "RLLM_OPENHANDS_TOP_P": str(self.top_p),
            "RLLM_OPENHANDS_MAX_ITERATIONS": str(self.max_iterations),
            "RLLM_OPENHANDS_CONDENSER_MAX_SIZE": str(self.condenser_max_size),
            "RLLM_OPENHANDS_CONDENSER_KEEP_FIRST": str(self.condenser_keep_first),
            "RLLM_OPENHANDS_LITELLM_EXTRA_BODY": json.dumps(self.litellm_extra_body or {}),
        }
        if self.max_input_tokens is not None:
            env["RLLM_OPENHANDS_MAX_INPUT_TOKENS"] = str(self.max_input_tokens)
        if self.max_output_tokens is not None:
            env["RLLM_OPENHANDS_MAX_OUTPUT_TOKENS"] = str(self.max_output_tokens)
        return env

    def _read_outcome(self, sandbox: Sandbox) -> dict | None:
        try:
            raw = sandbox.exec(
                f"cat {shlex.quote(_OUTCOME_PATH)} 2>/dev/null || true",
                timeout=15,
                user=self.agent_user,
            ).strip()
        except Exception as exc:
            logger.debug("OpenHands outcome read failed: %s", exc)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception as exc:
            logger.debug("OpenHands outcome parse failed: %s", exc)
            return None
        if data.get("finished") is True:
            return {}
        message = data.get("message", "")
        failure_traceback = data.get("traceback")
        if failure_traceback:
            message = f"{message}\n\n{failure_traceback}" if message else failure_traceback
        return {
            "exception_type": data.get("exception_type") or "ConversationIncompleteError",
            "message": message,
        }

    def write_configs(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        prompt = self.render_prompt(task, env["RLLM_OPENHANDS_WORKDIR"])
        self._exec_agent(sandbox, self._heredoc_write(_DRIVER_PATH, _DRIVER_SCRIPT), env=env)
        self._exec_agent(sandbox, self._heredoc_write(_PROMPT_PATH, prompt), env=env)

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        del instruction, task, config
        log = shlex.quote(self.stdout_log_path)
        return f"set -o pipefail; {_VENV_DIR}/bin/python {shlex.quote(_DRIVER_PATH)} 2>&1 | tee {log}"
