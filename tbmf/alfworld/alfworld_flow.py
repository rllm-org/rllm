"""Multi-turn ALFWorld agent flow.

The agent performs household tasks (pick-and-place, heat, cool, clean, etc.)
in a text-based environment by emitting natural-language commands. The whole
TextWorld env loop lives in this file — no dependency on ``rllm.environments``.

Task metadata schema (see ``prepare_alfworld_data.py``)::

    {"game_file": str, "task_type": str, "task_id": str, "max_steps": int}

The game file path points to a .tw-pddl file which TextWorld loads to
instantiate the environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import sys
import threading
import time
from functools import partial
from typing import Callable, TypeVar

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)

# Ensure alfworld_pkg is importable (lives alongside this file)
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_T = TypeVar("_T")

_ENV_REGISTRY_LOCK = threading.Lock()
_ENV_ID_CACHE: dict[tuple[str, int], str] = {}
_ENV_SEMAPHORES: dict[int, asyncio.Semaphore] = {}
_TEXTWORLD_PATCH_LOCK = threading.Lock()
_TEXTWORLD_PATCHED = False

# ---------------------------------------------------------------------------
# Monkey-patch TextWorld parsers to be thread-safe.
#
# TatSu parsers use internal mutable state (_statestack) that is NOT
# thread-local. When multiple threads call _PARSER.parse() concurrently,
# the shared state causes IndexError/TypeError/FailedToken crashes.
# Fix: replace each module-level _parse_and_convert with a version that
# instantiates a fresh parser per call (TatSu parsers are lightweight).
# ---------------------------------------------------------------------------

def _patch_textworld_parsers():
    """Make TextWorld's TatSu-based parsers thread-safe.

    TatSu parsers use internal mutable state (_statestack) that is not
    thread-local. Use thread-local storage to give each thread its own
    parser instance, avoiding the overhead of creating one per call while
    still being safe under concurrent access.
    """
    global _TEXTWORLD_PATCHED

    with _TEXTWORLD_PATCH_LOCK:
        if _TEXTWORLD_PATCHED:
            return
        _TEXTWORLD_PATCHED = True

    _thread_local = threading.local()

    try:
        import textworld.envs.pddl.logic as pddl_logic_mod
        from textworld.envs.pddl.logic.parser import PddlLogicParser
        from textworld.envs.pddl.logic.model import PddlLogicModelBuilderSemantics

        _OrigModelConverter = pddl_logic_mod._ModelConverter

        def _thread_safe_pddl_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "pddl_parser", None)
            if parser is None:
                parser = PddlLogicParser(semantics=PddlLogicModelBuilderSemantics(), parseinfo=True)
                _thread_local.pddl_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigModelConverter().walk(model)

        pddl_logic_mod._parse_and_convert = _thread_safe_pddl_parse_and_convert
        logger.info("Patched textworld.envs.pddl.logic._parse_and_convert for thread safety")
    except Exception as e:
        logger.warning("Failed to patch pddl logic parser: %s", e)

    try:
        import textworld.envs.pddl.textgen as textgen_mod
        from textworld.envs.pddl.textgen.parser import CSGParser
        from textworld.envs.pddl.textgen.model import CSGModelBuilderSemantics

        _OrigConverter = textgen_mod._Converter

        def _thread_safe_csg_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "csg_parser", None)
            if parser is None:
                parser = CSGParser(semantics=CSGModelBuilderSemantics(), parseinfo=True)
                _thread_local.csg_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigConverter().walk(model)

        textgen_mod._parse_and_convert = _thread_safe_csg_parse_and_convert
        logger.info("Patched textworld.envs.pddl.textgen._parse_and_convert for thread safety")
    except Exception as e:
        logger.warning("Failed to patch CSG parser: %s", e)

    try:
        import textworld.logic as tw_logic_mod
        from textworld.logic.parser import GameLogicParser
        from textworld.logic.model import GameLogicModelBuilderSemantics

        _OrigLogicConverter = tw_logic_mod._ModelConverter

        def _thread_safe_logic_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "logic_parser", None)
            if parser is None:
                parser = GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True)
                _thread_local.logic_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigLogicConverter().walk(model)

        tw_logic_mod._parse_and_convert = _thread_safe_logic_parse_and_convert
        logger.info("Patched textworld.logic._parse_and_convert for thread safety")
    except Exception as e:
        logger.warning("Failed to patch game logic parser: %s", e)

SYSTEM_PROMPT = """\
You are a helpful assistant controlling a robot in a household environment. Your task is to complete household tasks by navigating rooms and interacting with objects.

Action Commands:
Your <action> must be one of the following, strictly following the command (argument) format.

## Navigation & Observation:
- look: Look around your current location to get more details.
- inventory: Check the object you are currently holding (you can only hold one).
- go to (receptacle): Move to a receptacle (e.g., table, fridge, sink).

## Interacting with Receptacles:
- open (receptacle): Open a receptacle.
- close (receptacle): Close a receptacle.

## Interacting with Objects:
- take (object) from (receptacle): Pick up an object from a receptacle.
- put (object) in/on (receptacle): Place the object you are holding into or onto a receptacle.
- examine (object): Examine an object closely to learn its properties.

## Changing Object States:
- heat (object) with (receptacle): Heat an object with a device (e.g., microwave).
- cool (object) with (receptacle): Cool an object with a device (e.g., fridge).
- clean (object) with (receptacle): Clean an object with a device (e.g., sink).
- slice (object) with (object): Slice an object using a sharp object (e.g., knife).

Important Rules
1. You must first navigate to a location before interacting with objects there
2. You can only hold one object at a time
3. Some objects need to be heated, cooled, or cleaned before placing them
4. Always check admissible commands to see what actions are currently valid

Response Format
Think through your approach step by step, then provide your action in the following format:
<think>
Your reasoning and analysis here...
</think>
```action
<your action here>
```

you must adhere to the response format strictly or your action will not be executed:
For example:
<think>
I need to go to the countertop to get the knife.
</think>
```action
go to countertop 1
```

Always provide exactly ONE action at a time.
"""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"```action\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_GENERIC_RE = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)


def parse_action(response: str) -> str | None:
    """Extract action string from the model response.

    Looks for ```action ... ``` blocks first, then falls back to generic ``` ... ```.
    Returns None if no valid action block is found.
    """
    matches = _ACTION_RE.findall(response)
    if matches:
        action = matches[-1].strip()
        if action:
            return action

    matches = _GENERIC_RE.findall(response)
    if matches:
        action = matches[-1].strip()
        # Skip if it looks like a thinking block or code
        if action and not action.startswith("<") and "\n" not in action:
            return action

    return None


# ---------------------------------------------------------------------------
# Environment initialization (inline, no Ray)
# ---------------------------------------------------------------------------

# fast_downward.pddl2sas() uses module-level global state (options, counters,
# stdout capture) that is fundamentally not thread-safe. Env init calls
# pddl2sas twice (load + reset). We serialize init while allowing concurrent
# env.step() calls (step never invokes fast_downward).
#
# The async lock is intentionally taken before submitting work to the default
# executor. If hundreds of rollouts submit _run_env_init at once and block on a
# threading.Lock inside to_thread, the default executor can be saturated by lock
# waiters before the first ready rollout can even open its LLM HTTP connection.
_ENV_INIT_THREAD_LOCK = threading.Lock()
_ENV_INIT_ASYNC_LOCKS: dict[int, asyncio.Lock] = {}


def _env_concurrency_limit() -> int:
    raw = os.environ.get("ALFWORLD_ENV_CONCURRENCY", "256")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid ALFWORLD_ENV_CONCURRENCY=%r; using 256", raw)
        return 256


def _get_env_semaphore() -> asyncio.Semaphore | None:
    limit = _env_concurrency_limit()
    if limit <= 0:
        return None

    loop_id = id(asyncio.get_running_loop())
    sem = _ENV_SEMAPHORES.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(limit)
        _ENV_SEMAPHORES[loop_id] = sem
    return sem


def _get_env_init_lock() -> asyncio.Lock:
    loop_id = id(asyncio.get_running_loop())
    lock = _ENV_INIT_ASYNC_LOCKS.get(loop_id)
    if lock is None:
        lock = asyncio.Lock()
        _ENV_INIT_ASYNC_LOCKS[loop_id] = lock
    return lock


async def _run_env_io(fn: Callable[..., _T], *args) -> _T:
    """Run blocking TextWorld work off the event loop.

    TextWorld load/reset/step are synchronous and can take long enough to
    serialize all async rollouts before the first LLM request. A bounded
    thread offload lets LLM calls begin as soon as each environment is ready.
    """
    sem = _get_env_semaphore()
    if sem is None:
        return await asyncio.to_thread(fn, *args)
    async with sem:
        return await asyncio.to_thread(fn, *args)


async def _run_env_init(fn: Callable[..., _T], *args) -> _T:
    """Run env initialization serialized via lock.

    fast_downward.pddl2sas() mutates module-level globals (options flags,
    sys.stdout, counters) making it unsafe under concurrent threads. The
    coroutine-level lock queues init attempts without occupying executor
    threads; the thread lock remains as a process-wide guard in case another
    event loop tries to initialize TextWorld at the same time. env.step() does
    NOT call fast_downward, so steps remain concurrent.
    """
    def _locked_fn(*a):
        with _ENV_INIT_THREAD_LOCK:
            return fn(*a)

    async with _get_env_init_lock():
        return await asyncio.to_thread(_locked_fn, *args)


def _unwrap_single(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _normalize_infos(infos: dict) -> dict:
    normalized = {}
    for key, value in infos.items():
        if key == "admissible_commands":
            if (
                isinstance(value, (list, tuple))
                and len(value) == 1
                and isinstance(value[0], (list, tuple))
            ):
                normalized[key] = list(value[0])
            else:
                normalized[key] = value
            continue

        normalized[key] = _unwrap_single(value)
    return normalized


def _get_registered_env_id(game_file: str, max_steps: int) -> str:
    import textworld
    import textworld.gym

    from alfworld_pkg.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredInfos,
    )

    game_file = os.path.abspath(game_file)
    cache_key = (game_file, max_steps)

    with _ENV_REGISTRY_LOCK:
        env_id = _ENV_ID_CACHE.get(cache_key)
        if env_id is not None:
            return env_id

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )
        wrappers = [partial(AlfredDemangler, shuffle=False), AlfredInfos]
        digest = hashlib.sha1(f"{game_file}:{max_steps}".encode("utf-8")).hexdigest()[:12]
        env_id = textworld.gym.register_game(
            game_file,
            request_infos,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=wrappers,
            name=f"alfworld-{digest}",
        )
        _ENV_ID_CACHE[cache_key] = env_id
        return env_id


def _init_textworld_env(game_file: str, max_steps: int = 50):
    """Initialize a TextWorld environment for a single game file.

    Returns (env, obs, info) after reset.
    """
    _patch_textworld_parsers()

    import textworld.gym

    if not os.path.exists(game_file):
        raise FileNotFoundError(f"Game file not found: {game_file}")

    env_id = _get_registered_env_id(game_file, max_steps)
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()

    observation = _unwrap_single(obs)
    infos = _normalize_infos(infos)

    admissible_commands = infos.get("admissible_commands", [])

    return env, observation, admissible_commands


def _env_step(env, action: str) -> tuple[str, bool, bool, list[str]]:
    """Execute one step in the TextWorld environment.

    Returns (observation, won, done, admissible_commands).
    """
    obs, _scores, dones, infos = env.step(action)

    observation = _unwrap_single(obs)
    done = _unwrap_single(dones)
    done = bool(done)

    infos = _normalize_infos(infos)

    won = bool(infos.get("won", False))
    admissible_commands = infos.get("admissible_commands", [])

    return observation, won, done, admissible_commands


def _close_env(env) -> None:
    env.close()


# ---------------------------------------------------------------------------
# AgentFlow
# ---------------------------------------------------------------------------


@rllm.rollout(name="alfworld")
async def alfworld_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive the ALFWorld TextWorld env with an LLM until termination."""
    meta = task.metadata or {}
    game_file = meta.get("game_file")
    if not game_file:
        raise ValueError("Task metadata must include 'game_file'")

    max_steps = int(meta.get("max_steps", 50))
    task_type = meta.get("task_type", "unknown")

    env = None
    env_step_s = 0.0
    uid = config.session_uid or task.id
    t = time.perf_counter()
    logger.info("alfworld rollout %s: env init start game=%s max_steps=%d", uid, os.path.basename(game_file), max_steps)
    env, initial_obs, admissible_commands = await _run_env_init(_init_textworld_env, game_file, max_steps)
    env_init_s = time.perf_counter() - t
    logger.info(
        "alfworld rollout %s: env init done in %.2fs admissible_commands=%d",
        uid,
        env_init_s,
        len(admissible_commands),
    )

    try:
        client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")

        # Build initial user message
        commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)
        initial_prompt = (
            f"Observation:\n{initial_obs}\n\n"
            f"Admissible commands:\n{commands_str}\n\n"
            f"Steps remaining: {max_steps}\n\n"
            f"What is your next action?"
        )

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt},
        ]

        steps: list[Step] = []
        won = False
        last_action: str | None = None

        sampling = {k: v for k, v in config.sampling_params.items() if k != "top_k"}

        for turn in range(max_steps):
            try:
                if turn == 0:
                    logger.info("alfworld rollout %s: first LLM request start", uid)
                resp = await client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    **sampling,
                    timeout=120,
                )
                if turn == 0:
                    logger.info("alfworld rollout %s: first LLM response received", uid)
            except Exception as e:
                logger.warning("alfworld task %s turn %d: LLM call failed: %s", task.id, turn, e)
                break

            content = resp.choices[0].message.content or ""
            action_str = parse_action(content)
            last_action = action_str

            messages.append({"role": "assistant", "content": content})
            steps.append(
                Step(
                    chat_completions=list(messages),
                    model_response=content,
                    action=action_str,
                    thought=content,
                )
            )

            if action_str is None:
                # Invalid response — ask again
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last response did not contain a valid action. "
                        "Please reply with your action inside ```action ... ``` blocks.\n\n"
                        f"Admissible commands:\n{commands_str}\n\n"
                        "What is your next action?"
                    ),
                })
                continue

            # Execute action in environment
            t = time.perf_counter()
            observation, won, done, admissible_commands = await _run_env_io(_env_step, env, action_str)
            env_step_s += time.perf_counter() - t

            if done:
                break

            # Build next user prompt
            commands_str = "\n".join(f"  - {cmd}" for cmd in admissible_commands)
            remaining = max_steps - (turn + 1)
            user_msg = f"Observation:\n{observation}\n"

            if admissible_commands:
                user_msg += f"\nAdmissible commands:\n{commands_str}\n"

            user_msg += f"\nSteps remaining: {remaining}"

            if "Nothing happens" in observation:
                user_msg += f"\nNote: Your last action '{action_str}' had no effect. Please try a different action."

            user_msg += "\n\nWhat is your next action?"
            messages.append({"role": "user", "content": user_msg})

    finally:
        if env is not None:
            try:
                await _run_env_io(_close_env, env)
            except Exception:
                pass

    return Episode(
        trajectories=[Trajectory(name="alfworld", steps=steps)],
        metrics={
            "time/env_init_s": env_init_s,
            "time/env_step_s": env_step_s,
        },
        artifacts={
            "won": won,
            "turns": len(steps),
            "last_action": last_action,
            "task_type": task_type,
        },
        is_correct=won,
    )
