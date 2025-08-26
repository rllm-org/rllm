from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout_engine import RolloutEngine
from rllm.workflows.workflow import TerminationReason, Workflow


def _extract_textish(x: Any) -> str:
    """Extract plain text from possibly nested delta/data structures."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "text" in x:
            return _extract_textish(x["text"])
        if "delta" in x:
            return _extract_textish(x["delta"])
        if "data" in x:
            return _extract_textish(x["data"])
        if str(x.get("type", "")).endswith(".delta") and "text" in x:
            return _extract_textish(x["text"])
        return ""
    if isinstance(x, (list, tuple)):
        return "".join(_extract_textish(y) for y in x)
    return str(x)


def _stringify(obj: Any, limit: int = 800) -> str:
    """Stringify objects defensively, truncating to a safe length."""
    try:
        s = json.dumps(obj, ensure_ascii=False) if isinstance(obj, (dict, list)) else str(obj)
    except Exception:
        s = str(obj)
    return s[:limit]


def _summarize_observation(content: Any, max_preview_chars: int = 800) -> Any:
    """
    Produce a lightweight observation object suitable for logging and RL.

    - If content is small (string length or small JSON), return it directly
    - Otherwise return a compact dict with a preview
    """
    try:
        # Strings: return truncated string
        if isinstance(content, str):
            return content[:max_preview_chars]
        # JSON-like: measure by string size
        if isinstance(content, (dict, list)):
            s = _stringify(content, limit=max_preview_chars)
            # If stringified content already fits in preview, return original JSON
            if len(s) < max_preview_chars:
                return content
            return {"type": "tool.result.truncated", "preview": s}
        # Fallback: stringify other types
        return _stringify(content, limit=max_preview_chars)
    except Exception:
        return _stringify(content, limit=max_preview_chars)


@dataclass
class StrandsEvent:
    """
    Minimal unified event type for Strands streaming.
    """

    type: str  # "TextDelta" | "ToolUse" | "ToolResult" | "Stop"
    text: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    toolUseId: Optional[str] = None
    content: Optional[Any] = None
    status: Optional[str] = None  # "success" | "error"
    final_text: Optional[str] = None
    logprob: Optional[float] = None


class StrandsWorkflow(Workflow):
    """
    Skeleton Workflow that will consume a Strands Agent event stream and
    package it into an rLLM Episode/Trajectory/Step structure.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        strands_session_factory,  # callable: (system_prompt, tools, **kw) -> session with .step(user_msg) -> AsyncIterator[StrandsEvent]
        system_prompt: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 8,
        reward_fn=None,
        **kwargs,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, **kwargs)
        self.session_factory = strands_session_factory
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_steps = max_steps
        self.reward_fn = reward_fn

    async def __call__(self, task: dict | str, uid: str, **kwargs) -> Episode:
        """
        Minimal placeholder that returns a single-step Episode with the
        system and user messages wired in. Full event-stream handling to be
        implemented.
        """
        # Normalize task to a string message for the initial chat
        task_text = task if isinstance(task, str) else str(task)

        chat: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_text},
        ]

        trajectory = Trajectory(steps=[], reward=0.0)

        # Prefer streaming from a Strands session when the factory is provided; otherwise do a single-turn generation
        if self.session_factory is not None:
            assistant_buffer = ""
            last_text = None
            steps = 0
            term_reason = TerminationReason.ENV_DONE
            seen_tool_use_ids: set[str] = set()
            verbose = str(os.getenv("RLLM_STRANDS_VERBOSE", "1")).lower() not in ("0", "false", "no")
            # Track latest observed args per tool use id for streaming logs and RL storage
            tool_args_by_id: Dict[str, str] = {}
            tool_args_raw_by_id: Dict[str, Any] = {}
            tool_step_index_by_id: Dict[str, int] = {}
            tool_args_snapshot_by_id: Dict[str, str] = {}
            tool_args_printed_len_by_id: Dict[str, int] = {}
            tool_result_completed_ids: set[str] = set()
            # Pretty log helpers
            def _log(line: str = "", end: str = "\n") -> None:
                if verbose:
                    print(line, end=end, flush=True)
            def _h(line: str) -> None:
                _log(f"[STRANDS] {line}")
            debug_events = str(os.getenv("RLLM_DEBUG_EVENTS", "0")).lower() not in ("0", "false", "no")

            # Open session and stream events
            session = await self.session_factory(system_prompt=self.system_prompt, tools=self.tools)
            try:
                # Support either .step(task_text) or .stream(task_text)
                if hasattr(session, "step"):
                    stream_iter = session.step(task_text)
                elif hasattr(session, "stream"):
                    stream_iter = session.stream(task_text)
                else:
                    raise RuntimeError("Strands session must define .step() or .stream()")

                async for ev in stream_iter:
                    # Normalize type string from event
                    if isinstance(ev, dict):
                        typ = (ev.get("type") or ev.get("event") or "").lower()
                    else:
                        typ = (getattr(ev, "type", None) or getattr(ev, "event", "") or "").lower()

                    if typ in ("textdelta", "text_delta", "delta", "data"):
                        if debug_events:
                            _h(f"event=TextDelta raw={_stringify(ev)[:200]}")
                        txt = _extract_textish(
                            (ev.get("text") if isinstance(ev, dict) else getattr(ev, "text", None))
                            or (ev.get("delta") if isinstance(ev, dict) else getattr(ev, "delta", None))
                            or (ev.get("data") if isinstance(ev, dict) else getattr(ev, "data", None))
                            or ev
                        )
                        if txt and txt != last_text:
                            assistant_buffer += txt
                            # Stream assistant tokens as they arrive (single line)
                            _log(txt, end="")
                            last_text = txt

                    elif typ in ("tooluse", "tool_use", "current_tool_use"):
                        if debug_events:
                            _h(f"event=ToolUse raw={_stringify(ev)[:200]}")
                        tu = (ev.get("current_tool_use") if isinstance(ev, dict) else None) or ev
                        args = (tu.get("input") if isinstance(tu, dict) else getattr(tu, "input", {}))
                        # Normalize snapshot string for streaming display (handle dict or partial string)
                        if isinstance(args, (dict, list)):
                            args_str = json.dumps(args, ensure_ascii=False)
                        else:
                            args_str = _stringify(args)
                        if assistant_buffer:
                            # End the streaming line before logging tool use
                            _log()
                            chat.append({"role": "assistant", "content": assistant_buffer})
                            assistant_buffer = ""
                        name = (tu.get("name") if isinstance(tu, dict) else getattr(tu, "name", None)) or "unknown_tool"
                        tuid = (
                            (tu.get("toolUseId") if isinstance(tu, dict) else getattr(tu, "toolUseId", None))
                            or (tu.get("id") if isinstance(tu, dict) else getattr(tu, "id", None))
                            or ""
                        )
                        tool_args_by_id[tuid] = args_str
                        tool_args_raw_by_id[tuid] = args
                        # Create a single action step per toolUse id and print a single buffering header
                        if tuid not in seen_tool_use_ids:
                            seen_tool_use_ids.add(tuid)
                            _h(f"toolUse id={tuid} name={name} (buffering args)")
                            chat.append({"role": "assistant", "content": f"[toolUse id={tuid}] {name} { args_str }"})
                            action = {
                                "type": "tool_use",
                                "name": name,
                                "args": args,
                                "tool_use_id": tuid,
                            }
                            step = Step(chat_completions=list(chat), action=action, observation=None, reward=None)
                            trajectory.steps.append(step)
                            tool_step_index_by_id[tuid] = len(trajectory.steps) - 1
                            tool_args_snapshot_by_id[tuid] = ""
                            tool_args_printed_len_by_id[tuid] = 0
                        # Buffer snapshot only; do not print arg tokens incrementally
                        tool_args_snapshot_by_id[tuid] = args_str

                    elif typ in ("toolresult", "tool_result"):
                        if debug_events:
                            _h(f"event=ToolResult raw={_stringify(ev)[:200]}")
                        tuid = (
                            (ev.get("toolUseId") if isinstance(ev, dict) else getattr(ev, "toolUseId", None))
                            or (ev.get("tool_use_id") if isinstance(ev, dict) else getattr(ev, "tool_use_id", None))
                            or ""
                        )
                        status = (ev.get("status") if isinstance(ev, dict) else getattr(ev, "status", None)) or "success"
                        content = ev.get("content") if isinstance(ev, dict) else getattr(ev, "content", None)
                        # Print consolidated toolUse summary with final args, then the result status
                        # Emit final args once here to avoid noisy incremental logs
                        final_args_str = tool_args_by_id.get(tuid, "")
                        _h(f"toolUse id={tuid} name={tuid and (trajectory.steps[tool_step_index_by_id[tuid]].action.get('name') if tool_step_index_by_id.get(tuid) is not None else name)} final_args={final_args_str}")
                        _h(f"toolResult id={tuid} status={status}")
                        tool_result_completed_ids.add(tuid)
                        # Update the previously created tool_use step with final args for RL consumers
                        try:
                            idx = tool_step_index_by_id.get(tuid)
                            if idx is not None and 0 <= idx < len(trajectory.steps):
                                use_step = trajectory.steps[idx]
                                # Update action args with the latest raw args
                                final_args_raw = tool_args_raw_by_id.get(tuid)
                                if final_args_raw is not None:
                                    use_step.action["args"] = final_args_raw
                                # Update ALL steps' assistant toolUse message content to reflect final args
                                for s in trajectory.steps:
                                    for msg in s.chat_completions:
                                        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and msg["content"].startswith(f"[toolUse id={tuid}]"):
                                            tool_name_token = (msg['content'].split('] ', 1)[1].split(' ', 1)[0] if '] ' in msg['content'] else 'tool')
                                            msg["content"] = f"[toolUse id={tuid}] { tool_name_token } { final_args_str }"
                        except Exception:
                            pass
                        chat.append({"role": "user", "content": f"[toolResult id={tuid} status={status}] { _stringify(content) }"})
                        action = {"type": "tool_result", "tool_use_id": tuid, "status": status}
                        observation = _summarize_observation(content)
                        step = Step(chat_completions=list(chat), action=action, observation=observation, reward=None)
                        trajectory.steps.append(step)
                        steps += 1
                        if steps >= self.max_steps:
                            term_reason = TerminationReason.MAX_STEPS
                            break

                    elif typ in ("stop",):
                        if debug_events:
                            _h(f"event=Stop raw={_stringify(ev)[:200]}")
                        # Flush any toolUse entries that never emitted a ToolResult
                        try:
                            pending_ids = [pid for pid in seen_tool_use_ids if pid not in tool_result_completed_ids]
                            for pid in pending_ids:
                                final_args_str = tool_args_snapshot_by_id.get(pid) or tool_args_by_id.get(pid, "")
                                if final_args_str:
                                    tool_name = None
                                    idx = tool_step_index_by_id.get(pid)
                                    if idx is not None and 0 <= idx < len(trajectory.steps):
                                        tool_name = trajectory.steps[idx].action.get("name")
                                    _h(f"toolUse id={pid} name={tool_name or 'tool'} final_args={final_args_str}")
                                    # Update the step with best-effort final args
                                    try:
                                        if idx is not None and 0 <= idx < len(trajectory.steps):
                                            use_step = trajectory.steps[idx]
                                            raw = tool_args_raw_by_id.get(pid)
                                            if raw is None and isinstance(final_args_str, str) and final_args_str:
                                                try:
                                                    raw = json.loads(final_args_str)
                                                except Exception:
                                                    raw = final_args_str
                                            use_step.action["args"] = raw
                                        # Also update ALL steps' assistant toolUse message content to reflect final args
                                        for s in trajectory.steps:
                                            for msg in s.chat_completions:
                                                if msg.get("role") == "assistant" and isinstance(msg.get("content"), str) and msg["content"].startswith(f"[toolUse id={pid}]"):
                                                    tool_name_token = (msg['content'].split('] ', 1)[1].split(' ', 1)[0] if '] ' in msg['content'] else 'tool')
                                                    msg["content"] = f"[toolUse id={pid}] { tool_name_token } { final_args_str }"
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        final_text = (
                            (ev.get("final_text") if isinstance(ev, dict) else getattr(ev, "final_text", None))
                            or _extract_textish((ev.get("result") if isinstance(ev, dict) else getattr(ev, "result", None)))
                            or assistant_buffer
                        )
                        # Ensure we end the line for readability
                        _log()
                        if final_text:
                            chat.append({"role": "assistant", "content": final_text})
                            assistant_buffer = ""
                        if not trajectory.steps or trajectory.steps[-1].chat_completions != chat:
                            action = {"type": "final", "text": final_text or ""}
                            trajectory.steps.append(Step(chat_completions=list(chat), action=action, reward=None, done=True))
                        break

                episode = Episode(
                    id=uid,
                    task=task,
                    trajectories=[("solver", trajectory)],
                    is_correct=None,
                    metrics={"num_steps": len(trajectory.steps)},
                )
                episode.termination_reason = term_reason
                return episode

            finally:
                try:
                    await session.close()
                except Exception:
                    pass

        # Fallback: minimal single-turn generation so users can see a reply
        try:
            print("========= Fallback ==========")
            assistant_text = await self.rollout_engine.get_model_response(chat)
            if isinstance(assistant_text, str) and assistant_text:
                chat.append({"role": "assistant", "content": assistant_text})
        except Exception:
            pass

        trajectory.steps.append(Step(chat_completions=list(chat)))

        # Optionally compute a terminal reward via reward_fn
        if self.reward_fn is not None:
            try:
                trajectory.reward = float(
                    self.reward_fn(trajectory=trajectory, final_text="", steps=len(trajectory.steps))
                )
            except Exception:
                trajectory.reward = 0.0

        episode = Episode(
            id=uid,
            task=task,
            trajectories=[("solver", trajectory)],
            is_correct=None,
            metrics={"num_steps": len(trajectory.steps)},
        )
        episode.termination_reason = TerminationReason.ENV_DONE
        return episode


