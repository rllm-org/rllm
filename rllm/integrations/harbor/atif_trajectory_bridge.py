"""ATIF → rLLM trajectory bridge.

Converts Harbor's ATIF (Agent Trajectory Interchange Format) trajectory JSON
into rLLM training Step objects for episode logging and debugging in the eval
path. Token-level data (prompt_ids, response_ids, logprobs) is left empty —
those are collected via rLLM's gateway during training.

The public entry point is ``load_atif_steps(trial_uri)``.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rllm.types import Step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URI / filesystem helpers
# ---------------------------------------------------------------------------


def _file_uri_to_path(uri: str) -> Path:
    """Convert a ``file://`` URI to a local filesystem :class:`Path`."""
    parsed = urlparse(uri)
    # On Windows urlparse may leave a leading slash before the drive letter;
    # Path normalises that automatically.
    return Path(unquote(parsed.path))


# ---------------------------------------------------------------------------
# ATIF loading
# ---------------------------------------------------------------------------


def _load_atif_chain(agent_dir: Path) -> list[dict]:
    """Load the main trajectory and follow ``continued_trajectory_ref`` links.

    Returns a flat, ordered list of all ATIF step dicts across all chained
    trajectory files.
    """
    all_steps: list[dict] = []
    current_file = agent_dir / "trajectory.json"

    while current_file.exists():
        try:
            data = json.loads(current_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read ATIF trajectory %s: %s", current_file, exc)
            break

        steps = data.get("steps")
        if isinstance(steps, list):
            all_steps.extend(steps)

        # Follow the continuation chain if present.
        continued_ref = data.get("continued_trajectory_ref")
        if not continued_ref:
            break
        next_file = agent_dir / continued_ref
        if not next_file.exists():
            logger.warning(
                "ATIF continuation ref '%s' in %s does not exist",
                continued_ref,
                current_file.name,
            )
            break
        current_file = next_file

    return all_steps


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


def _normalize_content(content: Any) -> str:
    """Extract text from an ATIF message content field.

    Handles plain strings and ``list[ContentPart]`` (ATIF-v1.6 multimodal).
    Image parts are replaced with a ``[image: <path>]`` placeholder.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            elif ptype == "image":
                source = part.get("source") or {}
                path = source.get("path", "unknown")
                parts.append(f"[image: {path}]")
            else:
                parts.append(str(part))
        return "\n".join(parts)
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


# ---------------------------------------------------------------------------
# ATIF step → OpenAI chat messages
# ---------------------------------------------------------------------------


def _atif_step_to_chat_messages(step: dict) -> list[dict[str, str]]:
    """Convert one ATIF step dict to one or more OpenAI-format chat messages.

    Returns a list because an agent step with observations produces both an
    assistant message and a subsequent user message (the observation).

    The conversion matches the format used by ``traces_utils.py``
    (``_extract_single_episode_conversation``) so that SFT training data and
    the bridge produce consistent chat histories.
    """
    source = step.get("source")
    message = _normalize_content(step.get("message", ""))

    if source in ("system", "user"):
        return [{"role": "user", "content": message}]

    if source == "agent":
        # Build assistant content: <think>...</think> + message + <tool_call>s
        content_parts: list[str] = []

        reasoning = step.get("reasoning_content")
        if reasoning:
            content_parts.append(f"<think>{reasoning}</think>")

        if message:
            # Fix orphaned </think> tags (traces_utils.py:660-661 pattern)
            if "</think>" in message and "<think>" not in message:
                message = "<think>" + message
            content_parts.append(message)

        tool_calls = step.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tc_obj = {
                    "name": tc.get("function_name"),
                    "arguments": tc.get("arguments", {}),
                }
                tc_json = json.dumps(tc_obj, ensure_ascii=False)
                content_parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")

        assistant_content = "\n".join(content_parts) if content_parts else ""
        messages: list[dict[str, str]] = [{"role": "assistant", "content": assistant_content}]

        # Observation results → user message (feeds the next turn's context).
        observation = step.get("observation")
        if observation and isinstance(observation, dict):
            results = observation.get("results", [])
            obs_parts: list[str] = []
            for result in results:
                if isinstance(result, dict) and "content" in result:
                    obs_parts.append(_normalize_content(result["content"]))
            if obs_parts:
                messages.append({"role": "user", "content": "\n".join(obs_parts)})

        return messages

    # Unknown source — skip.
    return []


# ---------------------------------------------------------------------------
# Build a single rLLM training Step
# ---------------------------------------------------------------------------


def _build_step(
    atif_step: dict,
    chat_completions: list[dict[str, str]],
    is_last: bool,
) -> Step:
    """Create an rLLM training ``Step`` from an ATIF agent step.

    Args:
        atif_step: The raw ATIF step dict (``source="agent"``).
        chat_completions: Cumulative OpenAI-format messages *including* this
            step's assistant response.
        is_last: Whether this is the final emitted step.
    """
    # Tool calls
    action = None
    raw_tool_calls = atif_step.get("tool_calls")
    if raw_tool_calls:
        action = [
            {
                "name": tc.get("function_name"),
                "arguments": tc.get("arguments", {}),
            }
            for tc in raw_tool_calls
        ]

    # Observation text
    obs_text = None
    observation = atif_step.get("observation")
    if observation and isinstance(observation, dict):
        results = observation.get("results", [])
        obs_parts: list[str] = []
        for result in results:
            if isinstance(result, dict) and "content" in result:
                obs_parts.append(_normalize_content(result["content"]))
        if obs_parts:
            obs_text = "\n".join(obs_parts)

    # Metrics subset for metadata
    atif_metrics: dict[str, Any] | None = None
    raw_metrics = atif_step.get("metrics")
    if raw_metrics and isinstance(raw_metrics, dict):
        atif_metrics = {
            k: v
            for k, v in raw_metrics.items()
            if k
            in (
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "cost_usd",
            )
            and v is not None
        }
        if not atif_metrics:
            atif_metrics = None

    metadata: dict[str, Any] = {
        "atif_step_id": atif_step.get("step_id"),
        "source": "agent",
    }
    model_name = atif_step.get("model_name")
    if model_name:
        metadata["model_name"] = model_name
    timestamp = atif_step.get("timestamp")
    if timestamp:
        metadata["timestamp"] = timestamp
    if atif_metrics:
        metadata["atif_metrics"] = atif_metrics

    return Step(
        chat_completions=deepcopy(chat_completions),
        thought=atif_step.get("reasoning_content") or "",
        model_response=_normalize_content(atif_step.get("message", "")),
        action=action,
        observation=obs_text,
        reward=0.0,
        done=is_last,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_atif_steps(trial_uri: str) -> list[Step]:
    """Load an ATIF trajectory from disk and convert to rLLM training Steps.

    Each ATIF ``source="agent"`` step (one LLM generation) becomes one rLLM
    ``Step`` with cumulative ``chat_completions`` in OpenAI format.  Steps
    marked ``is_copied_context`` contribute to the chat history but are not
    emitted as training Steps.

    Token-level fields (``prompt_ids``, ``response_ids``, ``logprobs``) are
    left at their default empty values — those are collected via rLLM's
    gateway during training.

    Args:
        trial_uri: ``file://`` URI pointing to the trial directory.  The ATIF
            trajectory is expected at ``{trial_dir}/agent/trajectory.json``.

    Returns:
        List of rLLM training Steps, one per non-copied ATIF agent step.
        Returns ``[]`` if the trajectory does not exist or is malformed.
    """
    trial_dir = _file_uri_to_path(trial_uri)
    agent_dir = trial_dir / "agent"

    if not agent_dir.exists():
        logger.debug("No agent directory at %s", agent_dir)
        return []

    atif_steps = _load_atif_chain(agent_dir)
    if not atif_steps:
        logger.debug("No ATIF steps found under %s", agent_dir)
        return []

    # Walk through all steps, accumulating context and emitting Steps for
    # each non-copied agent turn.
    chat_completions: list[dict[str, str]] = []
    emitted: list[Step] = []

    # Pre-scan to find the last non-copied agent step index (for done=True).
    last_agent_idx = -1
    for i, step in enumerate(atif_steps):
        if step.get("source") == "agent" and not step.get("is_copied_context"):
            last_agent_idx = i

    for i, atif_step in enumerate(atif_steps):
        source = atif_step.get("source")

        if source in ("system", "user"):
            # Context — append to accumulator.
            chat_completions.extend(_atif_step_to_chat_messages(atif_step))
            continue

        if source == "agent":
            messages = _atif_step_to_chat_messages(atif_step)
            if not messages:
                continue

            # The first message is the assistant response.
            assistant_msg = messages[0]
            chat_completions.append(assistant_msg)

            # Emit a training Step only for non-copied agent steps.
            if not atif_step.get("is_copied_context"):
                is_last = i == last_agent_idx
                emitted.append(_build_step(atif_step, chat_completions, is_last))

            # Observation messages (if any) become context for the next turn.
            for msg in messages[1:]:
                chat_completions.append(msg)
            continue

        # Unknown source — add raw content as context if possible.
        message = _normalize_content(atif_step.get("message", ""))
        if message:
            chat_completions.append({"role": "user", "content": message})

    return emitted
