"""Terminus-2 with native tool-calling: same terminal loop, persistent reasoning.

Identical interaction model to :mod:`rllm.harnesses.terminus2` — the agent
drives a tmux session with batches of keystrokes and sees the terminal screen
after each batch — but the conversation rides the **native tool-calling rails**
instead of user-message simulation:

* the model emits an ``execute_commands`` tool call (same JSON schema Terminus
  asks for in prose);
* terminal output returns as a ``role:tool`` message;
* the model's ``reasoning_content`` is echoed back verbatim on each assistant
  message.

Why: reasoning models with separated thinking (e.g. DeepSeek-V4-Pro) only
retain reasoning across turns on the tool-calling context path. Measured on
Fireworks (2026-07-04): ``reasoning_content`` echoed in a user-message-style
history renders to zero prompt tokens (dropped), while the identical content on
an assistant message with ``tool_calls`` renders fully (+2.8k prompt tokens for
2.8k tokens of reasoning). DeepSeek's V4 report says the same in prose and
recommends non-think models for Terminus-style scaffolds; this harness makes
think modes first-class instead. Everything else — install, sentinel protocol,
budgets, gateway/session plumbing — is inherited unchanged.
"""

from __future__ import annotations

import os

from rllm.harnesses.terminus2 import (
    _DRIVER_PATH,
    _INSTRUCTION_PATH,
    Terminus2Harness,
)
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Task


class Terminus2NativeHarness(Terminus2Harness):
    """Terminus-2 loop over native tool-calling (reasoning persists across turns)."""

    name = "terminus2-native"

    # ---- Summarization knobs (harbor terminus_2.py parity) ----
    # Harbor reads the context limit from litellm's model map, which silently
    # assumes ~1M for unmapped models and never summarizes. Here it's explicit:
    # set per model (e.g. 240_000 for deepseek v4 on Fireworks); the driver
    # falls back to a conservative 350k-token default when unset.
    context_limit_tokens: int | None = None
    enable_summarize: bool = True
    # Free tokens below which the driver proactively summarizes (harbor
    # default 8000; 0 disables the proactive path, reactive still applies).
    proactive_summarization_threshold: int = 8000

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        env = super().build_env(task, config)
        env["RLLM_TERMINUS_ENABLE_SUMMARIZE"] = "1" if self.enable_summarize else "0"
        env["RLLM_TERMINUS_PROACTIVE_THRESHOLD"] = str(self.proactive_summarization_threshold)
        # Host env wins over the class attribute so eval runs can pin the
        # model's real limit (e.g. 240000 for v4 on Fireworks) without a code
        # edit — the eval CLI has no agent-kwargs mechanism.
        limit = os.environ.get("RLLM_TERMINUS_CONTEXT_LIMIT") or (str(self.context_limit_tokens) if self.context_limit_tokens is not None else None)
        if limit:
            env["RLLM_TERMINUS_CONTEXT_LIMIT"] = limit
        return env

    def write_configs(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        env: dict[str, str],
    ) -> None:
        instruction = str(task.instruction).strip()
        self._exec_agent(sandbox, self._heredoc_write(_DRIVER_PATH, _NATIVE_DRIVER_SCRIPT), env=env)
        self._exec_agent(sandbox, self._heredoc_write(_INSTRUCTION_PATH, instruction), env=env)


# ---------------------------------------------------------------------------
# In-sandbox driver. Reuses harbor's TmuxSession for terminal interaction and
# the terminus2 sentinel/env-var protocol, but owns the message loop: raw
# OpenAI-compatible HTTP to the gateway with tools + reasoning_content echo.
# ---------------------------------------------------------------------------
_NATIVE_DRIVER_SCRIPT = r'''
import asyncio
import json
import logging
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path, PurePosixPath

import httpx

from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType


class AgentSetupTimeoutError(asyncio.TimeoutError):
    pass


class AgentTimeoutError(asyncio.TimeoutError):
    pass


class ContextLengthExceededError(RuntimeError):
    """Raised immediately on a provider context-length error — retrying the
    same oversized payload would just fail identically, so llm() skips its
    retry loop and lets the caller summarize instead."""


log = logging.getLogger("terminus2-native-driver")

MAX_TOOL_OUTPUT_CHARS = 12000
# ~350k tokens at ~4 chars/token — leave headroom below the model's context.
MAX_HISTORY_CHARS = 1_400_000
KEEP_RECENT_TOOL_MSGS = 20

# Harbor-parity summarization knobs (harbor terminus_2.py defaults). Token
# counts are estimated (usage-anchored + chars/4) since litellm's counter isn't
# available in-sandbox; the context limit comes from env because litellm's
# model-map lookup is exactly what silently breaks on unmapped models.
DEFAULT_CONTEXT_LIMIT_TOKENS = 350_000
DEFAULT_PROACTIVE_THRESHOLD_TOKENS = 8000  # free tokens below which to summarize
UNWIND_TARGET_FREE_TOKENS = 4000

# Substrings seen across providers/gateways for "your prompt no longer fits".
CONTEXT_LENGTH_ERROR_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "prompt is too long",
    "reduce the length of the messages",
    "input length and `max_tokens`",
)

# The three-subagent summarization prompts, verbatim from harbor's
# terminus_2.py (including its rendered indentation), so the handoff content
# is identical to the original harness and eval deltas stay attributable to
# the transport alone.
SUMMARY_PROMPT_TEMPLATE = """You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""

QUESTION_PROMPT_TEMPLATE = """You are picking up work from a previous AI agent on this task:

**Original Task:** {original_instruction}

**Summary from Previous Agent:**
{summary}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the solution that are not answered in the summary
from the prior agent. After you ask these questions you will be on your own,
so ask everything you need to know."""

ANSWER_REQUEST_PREFIX = (
    "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n"
)

HANDOFF_PREFIX = "Here are the answers the other agent provided.\n\n"
HANDOFF_SUFFIX = (
    "\n\nContinue working on this task from where the previous agent left off."
    " You can no longer ask questions. Please follow the spec to interact with "
    "the terminal."
)

SHORT_SUMMARY_PROMPT_TEMPLATE = (
    "Briefly continue this task: {original_instruction}\n\n"
    "Current state: {limited_screen}\n\n"
    "Next steps (2-3 sentences):"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": (
                "Execute a batch of keystrokes in the task's tmux terminal and "
                "observe the resulting screen. Keystrokes are sent verbatim "
                "(end shell commands with \n). duration is how many seconds to "
                "wait after sending before the next command / screen capture."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string", "description": "What the current terminal state shows and what has been accomplished."},
                    "plan": {"type": "string", "description": "What the next commands are meant to achieve."},
                    "commands": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "keystrokes": {"type": "string"},
                                "duration": {"type": "number", "default": 1.0},
                            },
                            "required": ["keystrokes"],
                        },
                    },
                    "task_complete": {"type": "boolean", "default": False},
                },
                "required": ["commands"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "You are an AI assistant solving a command-line task in a Linux environment, "
    "operating a tmux terminal.\n\n"
    "Every turn you MUST call the execute_commands tool. Its keystrokes are sent to "
    "the terminal verbatim (remember trailing \n to run a shell command; durations "
    "let long commands finish before the screen is captured). The tool returns the "
    "new terminal output.\n\n"
    "Set task_complete=true when the task is fully done; you will be asked to call "
    "the tool once more with task_complete=true to confirm. Use commands=[] when "
    "you only want to re-observe the terminal or confirm completion."
)


class LocalEnvironment(BaseEnvironment):
    """BaseEnvironment whose exec runs on this machine (the sandbox itself)."""

    def __init__(self, workdir=None, session_id="terminus2-native-local"):
        self.session_id = session_id
        self.environment_name = session_id
        self.default_user = None
        self._workdir = workdir or None

    @staticmethod
    def type():
        return EnvironmentType.MODAL  # arbitrary; TmuxSession never branches on it

    @property
    def is_mounted(self):
        return True

    @property
    def supports_gpus(self):
        return False

    @property
    def can_disable_internet(self):
        return False

    def _validate_definition(self):
        return None

    async def start(self, force_build=False):
        return None

    async def stop(self, delete=False):
        return None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        run_cwd = cwd or self._workdir or None
        full_env = dict(os.environ)
        if env:
            full_env.update({k: str(v) for k, v in env.items()})
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=run_cwd,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return ExecResult(stdout="", stderr="command timed out", return_code=124)
        return ExecResult(
            stdout=out.decode("utf-8", "replace"),
            stderr=err.decode("utf-8", "replace"),
            return_code=proc.returncode if proc.returncode is not None else -1,
        )

    async def upload_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source_path), str(target_path))

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(str(source_dir), str(target_dir), dirs_exist_ok=True)

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source_path), str(target_path))

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(str(source_dir), str(target_dir), dirs_exist_ok=True)


def _truncate(text, limit=MAX_TOOL_OUTPUT_CHARS):
    if len(text) <= limit:
        return text
    head, tail = limit // 3, limit - limit // 3
    return text[:head] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-tail:]


def _history_chars(messages):
    return sum(
        len(m.get("content") or "") + len(m.get("reasoning_content") or "") + len(json.dumps(m.get("tool_calls") or []))
        for m in messages
    )


def _last_assistant_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return None


class Driver:
    def __init__(self):
        self.model = os.environ["RLLM_TERMINUS_MODEL"]
        if self.model.startswith("openai/"):
            self.model = self.model[len("openai/"):]  # gateway pins the model anyway
        self.api_base = (os.environ.get("RLLM_TERMINUS_API_BASE") or "").rstrip("/")
        self.temperature = float(os.environ.get("RLLM_TERMINUS_TEMPERATURE", "1.0"))
        _max_turns = os.environ.get("RLLM_TERMINUS_MAX_TURNS")
        self.max_steps = int(_max_turns) if _max_turns else 500
        self.enable_summarize = os.environ.get("RLLM_TERMINUS_ENABLE_SUMMARIZE", "1") == "1"
        self.proactive_threshold = int(os.environ.get("RLLM_TERMINUS_PROACTIVE_THRESHOLD", str(DEFAULT_PROACTIVE_THRESHOLD_TOKENS)))
        self.context_limit = int(os.environ.get("RLLM_TERMINUS_CONTEXT_LIMIT", str(DEFAULT_CONTEXT_LIMIT_TOKENS)))
        self.n_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.summarization_count = 0
        # Token estimate anchor: real usage from the last main-loop call plus
        # chars/4 for whatever was appended since. Reset whenever history is
        # rebuilt or unwound past the anchor point.
        self._anchor_tokens = 0
        self._anchor_idx = 0

    async def llm(self, client, messages, use_tools=True, tool_choice=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 16384,
            "temperature": self.temperature,
        }
        if use_tools:
            payload["tools"] = TOOLS
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        last_exc = None
        for attempt in range(4):
            if attempt:
                await asyncio.sleep(min(5 * 2 ** attempt, 40))
            try:
                r = await client.post(f"{self.api_base}/chat/completions", json=payload)
                if r.status_code != 200:
                    body = r.text
                    if any(marker in body.lower() for marker in CONTEXT_LENGTH_ERROR_MARKERS):
                        raise ContextLengthExceededError(f"gateway returned {r.status_code}: {body[:300]}")
                    last_exc = RuntimeError(f"gateway returned {r.status_code}: {body[:300]}")
                    log.warning("llm call attempt %d failed: %s", attempt + 1, last_exc)
                    continue
                data = r.json()
                if "choices" not in data:
                    # heartbeat-wrapped upstream error bodies arrive as 200 + {"error": ...}
                    body = json.dumps(data)
                    if any(marker in body.lower() for marker in CONTEXT_LENGTH_ERROR_MARKERS):
                        raise ContextLengthExceededError(f"upstream error body: {body[:300]}")
                    last_exc = RuntimeError(f"upstream error body: {body[:300]}")
                    log.warning("llm call attempt %d failed: %s", attempt + 1, last_exc)
                    continue
                self.n_calls += 1
                usage = data.get("usage") or {}
                self.in_tokens += usage.get("prompt_tokens") or 0
                self.out_tokens += usage.get("completion_tokens") or 0
                self.last_usage = usage
                return data["choices"][0]["message"]
            except ContextLengthExceededError:
                raise
            except Exception as e:  # noqa: BLE001 — retry transport blips
                last_exc = e
                log.warning("llm call attempt %d failed: %s", attempt + 1, e)
        raise RuntimeError(f"LLM call failed after 4 attempts: {last_exc}")

    def _estimate_tokens(self, messages):
        """Real usage from the last main call, plus chars/4 for anything
        appended since. Pure chars/4 when no anchor is valid."""
        if 0 < self._anchor_idx <= len(messages):
            return self._anchor_tokens + _history_chars(messages[self._anchor_idx:]) // 4
        return _history_chars(messages) // 4

    def _set_anchor(self, messages):
        usage = getattr(self, "last_usage", None) or {}
        self._anchor_tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
        self._anchor_idx = len(messages)

    def _reset_anchor(self):
        self._anchor_tokens = 0
        self._anchor_idx = 0

    def _unwind_messages_to_free_tokens(self, messages, target_free_tokens=UNWIND_TARGET_FREE_TOKENS):
        """Harbor parity: pop whole turns from the TAIL until the estimate
        leaves target_free_tokens headroom (harbor pops trailing message
        pairs). Keeps at least [system, first user]. Native twist: a turn is
        an assistant message plus its trailing tool results, popped as a unit
        so no orphaned tool message survives."""
        while len(messages) > 2:
            if self.context_limit - self._estimate_tokens(messages) >= target_free_tokens:
                break
            last_assistant = _last_assistant_index(messages)
            if last_assistant is None or last_assistant < 2:
                break
            del messages[last_assistant:]
        if self._anchor_idx > len(messages):
            self._reset_anchor()

    async def _summarize(self, client, messages, instruction, session):
        """Harbor's three-subagent summarization, verbatim flow:
        1. summary agent: full (unwound) history -> comprehensive summary;
        2. questions agent: FRESH context, sees task+summary+screen, asks what
           the summary doesn't answer;
        3. answers agent: full history + summary, answers those questions.
        History is then REBUILT to [system, question_prompt, questions] and
        the returned handoff (the answers) becomes the next user message."""
        self.summarization_count += 1

        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(original_instruction=instruction)
        smsg = await self.llm(client, messages + [{"role": "user", "content": summary_prompt}], tool_choice="none")
        summary = (smsg.get("content") or "").strip()
        if not summary:
            raise RuntimeError("summary subagent returned empty content")

        current_screen = await session.capture_pane(capture_entire=False)
        question_prompt = QUESTION_PROMPT_TEMPLATE.format(
            original_instruction=instruction, summary=summary, current_screen=current_screen
        )
        qmsg = await self.llm(client, [{"role": "user", "content": question_prompt}], use_tools=False)
        model_questions = (qmsg.get("content") or "").strip()
        if not model_questions:
            raise RuntimeError("questions subagent returned empty content")

        answers_history = messages + [
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": summary},
            {"role": "user", "content": ANSWER_REQUEST_PREFIX + model_questions},
        ]
        amsg = await self.llm(client, answers_history, tool_choice="none")
        answers = (amsg.get("content") or "").strip()
        if not answers:
            raise RuntimeError("answers subagent returned empty content")

        messages[:] = [
            messages[0],
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": model_questions},
        ]
        self._reset_anchor()
        return HANDOFF_PREFIX + answers + HANDOFF_SUFFIX

    async def _check_proactive_summarization(self, client, messages, instruction, session):
        """Harbor parity: at the top of each step, if estimated free tokens
        dip below the threshold, run the full summarization dance. Errors are
        logged and swallowed — the reactive path still backstops."""
        if not self.enable_summarize or self.proactive_threshold <= 0:
            return
        free_tokens = self.context_limit - self._estimate_tokens(messages)
        if free_tokens >= self.proactive_threshold:
            return
        log.info("Proactively summarizing. Free tokens: approximately %d", free_tokens)
        try:
            handoff = await self._summarize(client, messages, instruction, session)
            messages.append({"role": "user", "content": handoff})
        except Exception as e:  # noqa: BLE001 — harbor logs and continues
            log.error("Error in proactively summarizing: %s", e)

    async def _recover_from_context_overflow(self, client, messages, instruction, session):
        """Harbor's reactive fallback chain on ContextLengthExceededError:
        unwind to free tokens, then full summary -> short summary -> no-LLM
        handoff; retry the chat once and synthesize a response if even that
        fails (the episode must never crash here)."""
        self._unwind_messages_to_free_tokens(messages)

        summary_prompt = None
        try:
            log.debug("SUMMARIZATION: Attempting full summary")
            summary_prompt = await self._summarize(client, messages, instruction, session)
            log.info("SUMMARIZATION: Full summary succeeded")
        except Exception as e:  # noqa: BLE001 — fall through the tiers
            log.warning("SUMMARIZATION: Full summary failed: %s", e)

        if summary_prompt is None:
            try:
                log.debug("SUMMARIZATION: Attempting short summary")
                current_screen = await session.capture_pane(capture_entire=False)
                limited_screen = current_screen[-1000:] if current_screen else ""
                short_prompt = SHORT_SUMMARY_PROMPT_TEMPLATE.format(
                    original_instruction=instruction, limited_screen=limited_screen
                )
                short_msg = await self.llm(client, [{"role": "user", "content": short_prompt}], use_tools=False)
                short_summary = (short_msg.get("content") or "").strip()
                if not short_summary:
                    raise RuntimeError("short summarizer returned empty content")
                summary_prompt = f"{instruction}\n\nSummary: {short_summary}"
                log.info("SUMMARIZATION: Short summary succeeded")
            except Exception as e:  # noqa: BLE001 — last tier below never fails
                log.error("SUMMARIZATION: Short summary failed: %s", e)

        if summary_prompt is None:
            log.info("SUMMARIZATION: Using ultimate fallback")
            try:
                current_screen = await session.capture_pane(capture_entire=False)
            except Exception:  # noqa: BLE001
                current_screen = ""
            limited_screen = current_screen[-1000:] if current_screen else ""
            summary_prompt = f"{instruction}\n\nCurrent state: {limited_screen}"

        messages.append({"role": "user", "content": summary_prompt})
        try:
            return await self.llm(client, messages)
        except Exception as e:  # noqa: BLE001 — harbor fabricates a reply here
            log.error("Even fallback chat failed: %s", e)
            return {"content": "Technical difficulties. Please continue with the task."}

    def _enforce_history_budget(self, messages):
        total = _history_chars(messages)
        if total <= MAX_HISTORY_CHARS:
            return
        tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        for i in tool_idxs[:-KEEP_RECENT_TOOL_MSGS]:
            if messages[i]["content"] != "[truncated]":
                messages[i]["content"] = "[truncated]"
        # Old reasoning is the bulkiest component; drop it oldest-first too.
        assistant_idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("reasoning_content")]
        for i in assistant_idxs[:-KEEP_RECENT_TOOL_MSGS]:
            messages[i].pop("reasoning_content", None)

    async def run(self, instruction):
        env = LocalEnvironment(workdir=os.environ.get("RLLM_TERMINUS_WORKDIR") or None)
        pane_dir = Path(os.environ.get("RLLM_TERMINUS_LOGS_DIR", "/tmp/terminus2/logs"))
        try:
            pane_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pane_dir = Path("/tmp")
        session = TmuxSession(
            session_name="terminus-2",
            environment=env,
            logging_path=PurePosixPath(str(pane_dir / "terminus_2.pane")),
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            user=None,
        )
        await session.start()

        initial_screen = await session.get_incremental_output()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task:\n{instruction}\n\nCurrent terminal state:\n{_truncate(initial_screen)}"},
        ]
        pending_complete = False
        no_tool_strikes = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=60.0)) as client:
            for step in range(self.max_steps):
                await self._check_proactive_summarization(client, messages, instruction, session)
                try:
                    msg = await self.llm(client, messages)
                except ContextLengthExceededError as e:
                    if not self.enable_summarize:
                        log.warning("Context length exceeded and summarization is OFF.")
                        raise
                    log.warning("Context length exceeded. Using fallback summarization: %s", e)
                    msg = await self._recover_from_context_overflow(client, messages, instruction, session)
                tool_calls = msg.get("tool_calls") or []
                assistant = {"role": "assistant", "content": msg.get("content") or ""}
                if tool_calls:
                    assistant["tool_calls"] = tool_calls
                # THE point of this harness: echo reasoning so the tool-calling
                # context path retains it across turns.
                if msg.get("reasoning_content"):
                    assistant["reasoning_content"] = msg["reasoning_content"]
                messages.append(assistant)
                self._set_anchor(messages)

                if not tool_calls:
                    no_tool_strikes += 1
                    if no_tool_strikes >= 3:
                        log.warning("model stopped calling tools for %d turns; ending run", no_tool_strikes)
                        return
                    messages.append({"role": "user", "content": "Please call execute_commands (commands=[] just to observe, task_complete=true to finish)."})
                    continue
                no_tool_strikes = 0

                for tc in tool_calls:
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except (json.JSONDecodeError, TypeError) as e:
                        messages.append({"role": "tool", "tool_call_id": tc.get("id", "?"), "content": f"ERROR: could not parse tool arguments: {e}"})
                        continue
                    commands = args.get("commands") or []
                    timed_out = False
                    for cmd in commands:
                        keys = cmd.get("keystrokes") or ""
                        duration = float(cmd.get("duration") or 1.0)
                        try:
                            await session.send_keys(keys, block=False, min_timeout_sec=duration)
                        except Exception as e:  # noqa: BLE001 — mirror terminus2's per-command timeout reporting
                            timed_out = True
                            log.warning("send_keys failed: %s", e)
                            break
                    output = _truncate(await session.get_incremental_output())
                    if timed_out:
                        output += "\n[a command failed or timed out; terminal state above]"

                    if args.get("task_complete"):
                        if pending_complete:
                            log.info("task_complete confirmed at step %d", step)
                            return
                        pending_complete = True
                        output += (
                            "\n\nYou marked task_complete=true. If you are sure, call "
                            "execute_commands once more with task_complete=true (commands "
                            "may be empty). Otherwise continue working."
                        )
                    else:
                        pending_complete = False
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", "?"), "content": output})

                self._enforce_history_budget(messages)
            log.warning("max steps (%d) reached", self.max_steps)


async def _main():
    instruction = Path(os.environ["RLLM_TERMINUS_INSTRUCTION_FILE"]).read_text()
    logs_dir = Path(os.environ.get("RLLM_TERMINUS_LOGS_DIR", "/tmp/terminus2/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    agent_timeout = float(os.environ.get("RLLM_TERMINUS_AGENT_TIMEOUT_S", "0")) or None
    outcome_file = os.environ.get("RLLM_TERMINUS_OUTCOME_FILE")
    driver = Driver()
    log.info("terminus2-native-driver: model=%s max_steps=%s agent_timeout=%s", driver.model, driver.max_steps, agent_timeout)

    exc_info = None
    try:
        if agent_timeout:
            try:
                await asyncio.wait_for(driver.run(instruction), timeout=agent_timeout)
            except asyncio.TimeoutError:
                raise AgentTimeoutError(f"Agent execution timed out after {agent_timeout}s")
        else:
            await driver.run(instruction)
    except BaseException as e:  # noqa: BLE001 — record the verdict, never crash the exec
        exc_info = {
            "exception_type": type(e).__name__,
            "exception_message": str(e),
            "exception_traceback": traceback.format_exc(),
            "occurred_at": datetime.now().isoformat(),
        }
        log.warning("terminus2-native-driver agent phase failed: %s: %s", type(e).__name__, e)

    if outcome_file:
        try:
            Path(outcome_file).parent.mkdir(parents=True, exist_ok=True)
            Path(outcome_file).write_text(json.dumps({"exception_info": exc_info, "finished": exc_info is None}))
        except Exception as e:
            log.warning("terminus2-native-driver failed to write outcome file: %s", e)

    log.info(
        "terminus2-native-driver done: calls=%s in_tokens=%s out_tokens=%s exc=%s",
        driver.n_calls, driver.in_tokens, driver.out_tokens,
        (exc_info or {}).get("exception_type"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_main())
'''
