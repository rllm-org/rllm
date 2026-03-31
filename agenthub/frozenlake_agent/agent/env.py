"""Lightweight FrozenLake environment — no gymnasium dependency.

Reimplements the core grid-world logic from the legacy
rllm/environments/frozenlake/frozenlake.py using only numpy.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------


def _is_valid(board: list[list[str]], max_steps: int) -> bool:
    """DFS check that a path from S to G exists within max_steps."""
    arr = np.array(board)
    start_r, start_c = np.where(arr == "S")
    frontier: list[tuple[int, int, int]] = [(int(start_r[0]), int(start_c[0]), 0)]
    discovered: set[tuple[int, int]] = set()
    size = len(board)

    while frontier:
        r, c, steps = frontier.pop()
        if steps > max_steps:
            continue
        if (r, c) in discovered:
            continue
        discovered.add((r, c))
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if board[nr][nc] == "G":
                    return True
                if board[nr][nc] != "H":
                    frontier.append((nr, nc, steps + 1))
    return False


def generate_random_map(size: int = 8, p: float = 0.8, seed: int = 0, max_steps: int = 5) -> tuple[list[str], tuple[int, int]]:
    """Generate a random valid FrozenLake map.

    Args:
        size: Grid side length.
        p: Probability a tile is frozen (vs hole).
        seed: RNG seed for reproducibility.
        max_steps: Maximum steps for path-validity check.

    Returns:
        (map_rows, goal_position) where map_rows is a list of strings
        like ``["SFFF", "FHFH", "FFFH", "HFFG"]`` and goal_position
        is ``(row, col)`` of G.
    """
    rng = np.random.RandomState(seed)
    p = min(1.0, p)

    while True:
        board = rng.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()

        # Pick distinct start and goal positions
        while True:
            sr, sc = int(rng.randint(0, size)), int(rng.randint(0, size))
            gr, gc = int(rng.randint(0, size)), int(rng.randint(0, size))
            if (sr, sc) != (gr, gc):
                break

        board[sr][sc] = "S"
        board[gr][gc] = "G"

        if _is_valid(board, max_steps):
            return ["".join(row) for row in board], (gr, gc)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Action constants
ACTION_INVALID = 0
ACTION_LEFT = 1
ACTION_DOWN = 2
ACTION_RIGHT = 3
ACTION_UP = 4

ACTION_LOOKUP = {0: "None", 1: "Left", 2: "Down", 3: "Right", 4: "Up"}

# Deltas: (row_delta, col_delta) for each action
_DELTAS = {
    ACTION_LEFT: (0, -1),
    ACTION_DOWN: (1, 0),
    ACTION_RIGHT: (0, 1),
    ACTION_UP: (-1, 0),
}

# Render symbols
_GRID_LOOKUP = {
    "P": " P \t",
    "F": " _ \t",
    "H": " O \t",
    "G": " G \t",
    "X": " X \t",  # player fell in hole
    "V": " √ \t",  # player reached goal
}


class FrozenLakeEnv:
    """Pure-Python FrozenLake grid-world environment."""

    def __init__(
        self,
        size: int = 4,
        p: float = 0.8,
        seed: int = 42,
        max_steps: int = 5,
        is_slippery: bool = False,
    ):
        self.size = size
        self.p = p
        self.seed = seed
        self.max_steps = max_steps
        self.is_slippery = is_slippery

        self._map: list[str] = []
        self._goal: tuple[int, int] = (0, 0)
        self._player: tuple[int, int] = (0, 0)
        self._done = False
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Reset environment and return initial observation."""
        self._map, self._goal = generate_random_map(size=self.size, p=self.p, seed=self.seed, max_steps=self.max_steps)
        # Find start position
        for r, row in enumerate(self._map):
            for c, ch in enumerate(row):
                if ch == "S":
                    self._player = (r, c)
                    break
        self._done = False
        return self.render()

    def step(self, action: int) -> tuple[str, float, bool, dict]:
        """Take an action and return (observation, reward, done, info).

        Actions: 1=Left, 2=Down, 3=Right, 4=Up. 0 is invalid (no-op).
        """
        if self._done:
            return self.render(), 0.0, True, {"action_is_effective": False}

        if action == ACTION_INVALID or action not in _DELTAS:
            return self.render(), 0.0, False, {"action_is_effective": False}

        prev = self._player
        dr, dc = _DELTAS[action]
        nr, nc = prev[0] + dr, prev[1] + dc

        # Boundary check
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self._player = (nr, nc)

        tile = self._map[self._player[0]][self._player[1]]
        effective = self._player != prev

        if tile == "G":
            self._done = True
            return self.render(), 1.0, True, {"action_is_effective": effective}
        if tile == "H":
            self._done = True
            return self.render(), 0.0, True, {"action_is_effective": effective}

        return self.render(), 0.0, False, {"action_is_effective": effective}

    def render(self) -> str:
        """Render the grid as a text string (P=player, _=frozen, O=hole, G=goal)."""
        rows = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                if (r, c) == self._player:
                    tile = self._map[r][c]
                    if tile == "H":
                        cells.append(_GRID_LOOKUP["X"])
                    elif tile == "G":
                        cells.append(_GRID_LOOKUP["V"])
                    else:
                        cells.append(_GRID_LOOKUP["P"])
                else:
                    ch = self._map[r][c]
                    # Replace start marker with frozen
                    sym = "F" if ch == "S" else ch
                    cells.append(_GRID_LOOKUP[sym])
            rows.append("".join(cells))
        return "\n".join(rows)

    def finished(self) -> bool:
        return self._done

    def success(self) -> bool:
        return self._done and self._map[self._player[0]][self._player[1]] == "G"
