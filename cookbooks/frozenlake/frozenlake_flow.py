"""Multi-turn FrozenLake agent.

The agent navigates a randomly-generated FrozenLake grid by emitting
``Up``/``Down``/``Left``/``Right`` actions until it reaches the goal,
falls in a hole, or runs out of steps. The whole gym-env loop lives
in this file — no dependency on ``rllm.environments``.

Task metadata schema (see ``prepare_frozenlake_data.py``)::

    {"seed": int, "size": int, "p": float, "is_slippery": bool, "max_steps": int}

The map is regenerated deterministically from ``seed`` + ``size`` + ``p``
each time the flow runs, so episodes are reproducible.
"""

from __future__ import annotations

import logging
import re

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)


# Gymnasium FrozenLake-v1 native action codes.
_ACTIONS = {"left": 0, "down": 1, "right": 2, "up": 3}
_ACTION_LABELS = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


SYSTEM_PROMPT = """\
You are walking on a frozen lake. Reach the goal (G) without falling into any hole (O).

Symbols:
    P  = your current position
    _  = frozen tile (safe)
    O  = hole (terminal, you lose)
    G  = goal (terminal, you win)

Valid actions: Up | Down | Left | Right

On every turn, briefly explain your reasoning, then output the chosen action
inside triple backticks on its own line. Example:

    I should move down to avoid the hole on my right.
    ```Down```

Only the contents of the LAST set of triple backticks count as your action.
Output exactly one of: Up, Down, Left, Right.
"""


# ---------------------------------------------------------------------------
# Random map generation (adapted from gymnasium / RAGEN; kept self-contained
# so this cookbook has no dependency on rllm.environments).
# ---------------------------------------------------------------------------

_MAX_PATH = 50  # cap on DFS depth when validating reachability


def _is_solvable(board: list[list[str]], size: int) -> bool:
    """DFS check: is there a frozen path from S to G?"""
    start_r, start_c = np.where(np.array(board) == "S")
    if start_r.size == 0:
        return False
    frontier: list[tuple[int, int, int]] = [(int(start_r[0]), int(start_c[0]), 0)]
    seen: set[tuple[int, int]] = set()
    while frontier:
        r, c, steps = frontier.pop()
        if steps > _MAX_PATH or (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if board[nr][nc] == "G":
                return True
            if board[nr][nc] != "H":
                frontier.append((nr, nc, steps + 1))
    return False


def generate_random_map(size: int, p: float, seed: int) -> list[str]:
    """Generate a deterministically-seeded ``size x size`` map with a guaranteed S→G path."""
    rng, _ = seeding.np_random(seed)
    p = min(1.0, max(0.0, p))
    while True:
        board = rng.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()
        # Place S and G at distinct random tiles.
        for _ in range(64):
            sr, sc = int(rng.integers(0, size)), int(rng.integers(0, size))
            gr, gc = int(rng.integers(0, size)), int(rng.integers(0, size))
            if (sr, sc) != (gr, gc):
                break
        board[sr][sc] = "S"
        board[gr][gc] = "G"
        if _is_solvable(board, size):
            return ["".join(row) for row in board]


# ---------------------------------------------------------------------------
# Rendering + parsing
# ---------------------------------------------------------------------------


def render_grid(env: gym.Env) -> str:
    """Render the current state as a P/_/O/G grid (no S, since the player has moved)."""
    unwrapped = env.unwrapped
    desc = unwrapped.desc.astype(str)
    nrow, ncol = desc.shape
    pr, pc = divmod(int(unwrapped.s), ncol)
    rows: list[str] = []
    for r in range(nrow):
        cells = []
        for c in range(ncol):
            if (r, c) == (pr, pc):
                cells.append("P")
            else:
                tile = desc[r, c]
                cells.append({"S": "_", "F": "_", "H": "O", "G": "G"}.get(tile, tile))
        rows.append("  ".join(cells))
    return "\n".join(rows)


_ACTION_RE = re.compile(r"```(.*?)```", re.DOTALL)


def parse_action(response: str) -> int | None:
    """Extract a 0-3 action from the *last* ```...``` block in the response.

    Returns ``None`` if no valid action is found.
    """
    matches = _ACTION_RE.findall(response)
    if not matches:
        return None
    text = matches[-1].strip().lower()
    return _ACTIONS.get(text)


# ---------------------------------------------------------------------------
# AgentFlow
# ---------------------------------------------------------------------------


@rllm.rollout(name="frozenlake")
async def frozenlake_flow(task: Task, config: AgentConfig) -> Episode:
    """Drive the FrozenLake gym env with an LLM until termination."""
    meta = task.metadata or {}
    seed = int(meta.get("seed", 42))
    size = int(meta.get("size", 4))
    p = float(meta.get("p", 0.8))
    is_slippery = bool(meta.get("is_slippery", False))
    max_turns = int(meta.get("max_steps", max(2 * size, 16)))

    desc = generate_random_map(size=size, p=p, seed=seed)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
    env.reset(seed=seed)

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"Current observation (turn 0):\n{render_grid(env)}\n\nYou have {max_turns} turns total. Output your next action."),
        },
    ]

    steps: list[Step] = []
    won = False
    last_action_label: str | None = None

    for turn in range(max_turns):
        try:
            resp = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=1.0,
                max_tokens=2048,
                timeout=120,
            )
        except Exception as e:
            logger.warning("frozenlake task %s turn %d: LLM call failed: %s", task.id, turn, e)
            break

        content = resp.choices[0].message.content or ""
        action = parse_action(content)
        last_action_label = _ACTION_LABELS.get(action) if action is not None else None

        messages.append({"role": "assistant", "content": content})
        steps.append(
            Step(
                chat_completions=list(messages),
                model_response=content,
                action=last_action_label,
                thought=content,
            )
        )

        if action is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your last response did not contain a valid action. Please reply with one of Up, Down, Left, Right inside triple backticks.\nCurrent observation:\n{render_grid(env)}"
                    ),
                }
            )
            continue

        _, reward, terminated, truncated, _ = env.step(action)
        if terminated:
            won = float(reward) > 0
            break
        if truncated:
            break

        messages.append(
            {
                "role": "user",
                "content": (f"Current observation (turn {turn + 1}):\n{render_grid(env)}\nOutput your next action."),
            }
        )

    return Episode(
        trajectories=[Trajectory(name="frozenlake", steps=steps)],
        artifacts={"won": won, "turns": len(steps), "last_action": last_action_label},
        is_correct=won,
    )
