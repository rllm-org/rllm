"""Best-effort OpenHands profile for the llm-stats GLM-5.2 result.

llm-stats reports a 0.621 resolve rate for GLM-5.2 on SWE-bench Pro, but
labels every result on that page self-reported and publishes neither code nor
task-level traces.  Z.ai discloses only that its run used OpenHands, a tailored
prompt, temperature 1, top-p 1, 32K output tokens, and a 400K context window.

This harness therefore pins the current public OpenHands SWE-bench Pro
baseline and records the remaining provenance gap.  It is a reproduction
attempt, not a claim that Z.ai's undisclosed prompt and OpenHands revision have
been recovered.
"""

# ruff: noqa: E501 -- the vendored public prompt keeps upstream line breaks.

from __future__ import annotations

import json
import shlex

from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Episode, Task, TerminationReason

LLM_STATS_REFERENCE_URL = "https://llm-stats.com/benchmarks/swe-bench-pro"
LLM_STATS_TARGET_SCORE = 0.621
TARGET_MODEL = "GLM-5.2"
OPENROUTER_MODEL_ID = "z-ai/glm-5.2"
OPENROUTER_PROVIDER_SLUG = "z-ai"

# Public baseline inspected on 2026-08-08. The benchmark repo pins the Agent
# SDK through a git submodule; both revisions are kept here so an old rLLM run
# remains reconstructable even after either upstream default branch moves.
OPENHANDS_BENCHMARKS_REVISION = "1411fe96666e2c00b958cd30055ad232e2a64ca1"
OPENHANDS_SDK_REVISION = "43376f1868ffd702746080714a59c16d3f69ec12"
OPENHANDS_EXTENSIONS_REVISION = "2cfd75e2396ad5111f0e1670a5534a79bbf97a3e"

TEMPERATURE = 1.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 32_768
CONTEXT_WINDOW_TOKENS = 400_000
MAX_ITERATIONS = 500
CONDENSER_MAX_SIZE = 240
CONDENSER_KEEP_FIRST = 2

_VENV_DIR = "/opt/rllm/swebench-pro-openhands-glm52"
_DRIVER_PATH = "/tmp/rllm-swebench-pro-openhands.py"
_PROMPT_PATH = "/tmp/rllm-swebench-pro-openhands-prompt.txt"
_SUMMARY_PATH = "/tmp/rllm-swebench-pro-openhands-summary.json"
_REVISION_FILE = f"{_VENV_DIR}/.sdk-revision"
_SDK_ARCHIVE = f"https://github.com/OpenHands/software-agent-sdk/archive/{OPENHANDS_SDK_REVISION}.tar.gz"
_SDK_REQUIREMENT = f"openhands-sdk @ {_SDK_ARCHIVE}#subdirectory=openhands-sdk"
_TOOLS_REQUIREMENT = f"openhands-tools @ {_SDK_ARCHIVE}#subdirectory=openhands-tools"

REPRODUCTION_PROFILE = {
    "status": "best_effort",
    "score_source": LLM_STATS_REFERENCE_URL,
    "target_model": TARGET_MODEL,
    "target_score": LLM_STATS_TARGET_SCORE,
    "score_status": "self_reported_unverified",
    "openhands_benchmarks_revision": OPENHANDS_BENCHMARKS_REVISION,
    "openhands_sdk_revision": OPENHANDS_SDK_REVISION,
    "openhands_extensions_revision": OPENHANDS_EXTENSIONS_REVISION,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "context_window_tokens": CONTEXT_WINDOW_TOKENS,
    "max_iterations": MAX_ITERATIONS,
    "prompt_provenance": "public OpenHands SWE-bench Pro baseline; Z.ai tailored prompt undisclosed",
    "prompt_input": "rLLM official task instruction wrapped by the public OpenHands prompt",
    "workspace": "official task image /app exposed through OpenHands LocalWorkspace",
    "openrouter_provider_route": {
        "only": [OPENROUTER_PROVIDER_SLUG],
        "allow_fallbacks": False,
        "require_parameters": True,
    },
}

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
if [ "$installed_revision" != "{OPENHANDS_SDK_REVISION}" ]; then
    uv venv --python 3.12 --clear {_VENV_DIR}
    uv pip install --python {_VENV_DIR}/bin/python {shlex.quote(_SDK_REQUIREMENT)}
    uv pip install --python {_VENV_DIR}/bin/python --no-deps {shlex.quote(_TOOLS_REQUIREMENT)}
    uv pip install --python {_VENV_DIR}/bin/python binaryornot cachetools libtmux func-timeout
    printf '%s\n' "{OPENHANDS_SDK_REVISION}" > {_REVISION_FILE}
fi
"""


def _render_prompt(instruction: str, workdir: str, base_commit: str) -> str:
    """Render the pinned public OpenHands SWE-bench Pro prompt baseline."""
    return f"""I have access to a code repository in the directory {workdir} . You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{instruction}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way.
The benchmark image already includes the repository and its baseline dependencies, so prefer using the existing environment and only install missing dependencies when the repository clearly requires it.
Your task is to make the minimal changes to non-test files in the {workdir} directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets, explain any best practices or conventions they imply.
   1.2 Highlight error messages, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing the issue.

Phase 2. RUNNING: understand how the repository is built and tested
   2.1 Read the repository docs and relevant config files to understand the expected workflow.
   2.2 Identify the project language, package manager, test runner, and any required services.
   2.3 Run the most relevant tests or reproduction steps for this issue.

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use search tools to locate relevant methods, classes, keywords, and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the most likely files and functions to change, and explain why.
   3.4 Select the best fix location before editing.

Phase 4. TEST CREATION: before implementing any fix, create a script or command sequence to reproduce and verify the issue.
   4.1 Look at existing tests to understand the expected style and structure.
   4.2 Create a minimal reproduction that demonstrates the issue.
   4.3 Run it to confirm you are reproducing the problem.
   4.4 Refine it as needed.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the reproduction proves the issue.
   5.4 State clearly any best practices to preserve in the fix.
   5.5 State clearly how you will fix the problem.

Phase 6. FIX IMPLEMENTATION: edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: test your implementation thoroughly.
   7.1 Re-run your reproduction to verify the fix works.
   7.2 Add edge cases when useful.
   7.3 Run existing tests related to the modified code to ensure you have not broken anything else.

Phase 8. FINAL REVIEW: carefully re-read the problem description and compare your changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any relevant tests for the issue, the files you modified, and the functions you changed.
   8.3 If any tests fail, revise your implementation until all relevant tests pass.

Be thorough in your exploration, testing, and reasoning. It is fine if your thinking process is lengthy: quality and completeness are more important than brevity.
"""


_DRIVER_SCRIPT = rf'''"""Pinned local-workspace OpenHands driver for rLLM."""

import json
import os

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.context import AgentContext
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.skills import load_public_skills
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


def main():
    model = os.environ["LLM_MODEL"]
    provider_route = None
    if model.startswith("openrouter/"):
        provider_route = {{
            "provider": {{
                "only": ["{OPENROUTER_PROVIDER_SLUG}"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }}
        }}
    llm = LLM(
        usage_id="agent",
        model=model,
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature={TEMPERATURE},
        top_p={TOP_P},
        max_input_tokens={CONTEXT_WINDOW_TOKENS},
        max_output_tokens={MAX_OUTPUT_TOKENS},
        disable_vision=True,
        litellm_extra_body=provider_route or {{}},
    )
    condenser_llm = llm.model_copy(deep=True, update={{"usage_id": "condenser"}})
    public_skills = load_public_skills()
    agent_context = AgentContext(skills=public_skills) if public_skills else None
    agent = Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False),
        system_prompt_kwargs={{"cli_mode": True}},
        condenser=LLMSummarizingCondenser(
            llm=condenser_llm,
            max_size={CONDENSER_MAX_SIZE},
            keep_first={CONDENSER_KEEP_FIRST},
        ),
        agent_context=agent_context,
    )
    conversation = Conversation(
        agent=agent,
        workspace=os.environ["RLLM_OPENHANDS_WORKDIR"],
        max_iteration_per_run={MAX_ITERATIONS},
        delete_on_close=True,
    )
    instruction = open(os.environ["RLLM_OPENHANDS_PROMPT_PATH"], encoding="utf-8").read()
    conversation.send_message(instruction)

    fake_responses = 0
    timeout = int(os.environ.get("CONVERSATION_TIMEOUT", "3600"))
    while True:
        conversation.run(timeout=timeout)
        events = list(conversation.state.events)
        if conversation.state.execution_status != ConversationExecutionStatus.FINISHED:
            break
        if _finished(events) or not _sent_message(events) or fake_responses >= 10:
            break
        conversation.send_message(_fake_response(conversation))
        fake_responses += 1

    summary = {{
        "execution_status": conversation.state.execution_status.value,
        "events": len(list(conversation.state.events)),
        "fake_responses": fake_responses,
        "profile": {json.dumps(REPRODUCTION_PROFILE, sort_keys=True)},
    }}
    with open(os.environ["RLLM_OPENHANDS_SUMMARY_PATH"], "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
'''


class SwebenchProOpenHandsGLM52Harness(BaseCliHarness):
    """Run the public OpenHands baseline toward llm-stats' GLM-5.2 score."""

    name = "swebench-pro-openhands-glm52"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/swebench-pro-openhands-glm52.log"

    target_score = LLM_STATS_TARGET_SCORE
    max_iterations = MAX_ITERATIONS
    temperature = TEMPERATURE
    top_p = TOP_P
    max_output_tokens = MAX_OUTPUT_TOKENS
    context_window_tokens = CONTEXT_WINDOW_TOKENS

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        if "glm-5.2" not in config.model.lower():
            raise ValueError(f"{self.name} requires a GLM-5.2 model, got {config.model!r}")
        if config.model.lower().startswith("z-ai/"):
            # rLLM's OpenRouter proxy exposes the exact OpenRouter model ID as
            # its alias. Tell OpenHands/LiteLLM which provider syntax to use,
            # while preserving that alias in the forwarded request body.
            qualified = f"openrouter/{config.model}"
        else:
            _, _, qualified = self.ensure_provider_prefix(config.model)
        api_key = self.gateway_api_key(config, "OPENAI_API_KEY")
        timeout = int(float(task.metadata.get("agent_timeout") or self.run_timeout))
        workdir = str(task.metadata.get("workdir") or "/app")
        return {
            "LLM_MODEL": qualified,
            "LLM_API_KEY": api_key,
            "LLM_BASE_URL": config.base_url,
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_BASE": config.base_url,
            "OPENAI_BASE_URL": config.base_url,
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "EXTENSIONS_REF": OPENHANDS_EXTENSIONS_REVISION,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "CONVERSATION_TIMEOUT": str(timeout),
            "RLLM_OPENHANDS_WORKDIR": workdir,
            "RLLM_OPENHANDS_PROMPT_PATH": _PROMPT_PATH,
            "RLLM_OPENHANDS_SUMMARY_PATH": _SUMMARY_PATH,
        }

    def write_configs(self, sandbox: Sandbox, task: Task, config: AgentConfig, env: dict[str, str]) -> None:
        metadata = task.metadata.get("metadata", {}) or {}
        base_commit = str(task.metadata.get("base_commit") or metadata.get("base_commit") or "the task base commit")
        workdir = env["RLLM_OPENHANDS_WORKDIR"]
        prompt = _render_prompt(str(task.instruction).strip(), workdir, base_commit)
        self._exec_agent(sandbox, self._heredoc_write(_DRIVER_PATH, _DRIVER_SCRIPT), env=env)
        self._exec_agent(sandbox, self._heredoc_write(_PROMPT_PATH, prompt), env=env)

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        del instruction, task, config
        return f"set -o pipefail; {_VENV_DIR}/bin/python {shlex.quote(_DRIVER_PATH)} 2>&1 | tee {shlex.quote(self.stdout_log_path)}"

    def _outcome_episode(
        self,
        task: Task,
        termination_reason: TerminationReason | None = None,
        error: dict | None = None,
    ) -> Episode:
        episode = super()._outcome_episode(task, termination_reason, error)
        episode.metadata = dict(episode.metadata or {})
        episode.metadata["reproduction_profile"] = dict(REPRODUCTION_PROFILE)
        return episode
