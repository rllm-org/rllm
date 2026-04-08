"""Evolving hint pool for GSD on structurally homogeneous tasks.

For tasks where all problems share the same structure (e.g., countdown),
per-problem hints are wasteful.  Instead, this module maintains a pool of
**generic strategy hints** that are scored by teacher-student improvement
and periodically evolved via an external LLM (through LiteLLM + OpenRouter).

Usage::

    pool = HintPool(seed_hints=["- Try working backwards...", ...])

    # During training:
    hint = pool.select()                       # UCB1 selection
    ...run GSD loop...
    pool.update(hint, improvement=R_T - R_S)   # update EMA score

    # Periodically:
    if pool.should_evolve(step):
        await pool.evolve(hard_solves)
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model for hint evolution via LiteLLM + OpenRouter
DEFAULT_EVOLVE_MODEL = "openrouter/google/gemini-3-flash-preview"


@dataclass
class ScoredHint:
    """A hint candidate with a running performance score."""

    text: str
    score: float = 0.0  # EMA of teacher improvement when this hint was used
    n_uses: int = 0
    created_at: int = 0  # training step when created

    def to_dict(self) -> dict:
        return {"text": self.text, "score": self.score, "n_uses": self.n_uses, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: dict) -> ScoredHint:
        return cls(text=d["text"], score=d.get("score", 0.0), n_uses=d.get("n_uses", 0), created_at=d.get("created_at", 0))


class HintPool:
    """Pool of scored hint candidates with UCB1 selection and LLM-based evolution.

    Args:
        max_size: Maximum number of hints in the pool.
        ema_alpha: Smoothing factor for EMA score updates (higher = more reactive).
        ucb_c: Exploration coefficient for UCB1 selection.
        evolve_every: Evolve the pool every N training steps (0 = never).
        evolve_model: LiteLLM model identifier for hint evolution
            (default: ``openrouter/google/gemini-2.5-flash-preview``).
        seed_hints: Initial hint texts to populate the pool.
        save_path: Path for JSON persistence (None = no saving).
        autosave_every: Save after every N updates (0 = never).
    """

    def __init__(
        self,
        max_size: int = 5,
        ema_alpha: float = 0.3,
        ucb_c: float = 1.0,
        evolve_every: int = 20,
        evolve_model: str = DEFAULT_EVOLVE_MODEL,
        seed_hints: list[str] | None = None,
        save_path: str | Path | None = None,
        autosave_every: int = 10,
    ) -> None:
        self.max_size = max_size
        self.ema_alpha = ema_alpha
        self.ucb_c = ucb_c
        self.evolve_every = evolve_every
        self.evolve_model = evolve_model
        self._save_path = Path(save_path) if save_path is not None else None
        self._autosave_every = autosave_every
        self._hints: list[ScoredHint] = []
        self._total_uses = 0
        self._updates_since_save = 0
        self._last_evolve_step = 0
        self._hard_solves: list[dict] = []  # shared across all workflow instances
        self._max_hard_solves = 20
        self._evolving = False  # guard against concurrent evolve calls

        if seed_hints:
            for text in seed_hints:
                self._hints.append(ScoredHint(text=text))

    @property
    def size(self) -> int:
        return len(self._hints)

    # ------------------------------------------------------------------
    # Selection (UCB1)
    # ------------------------------------------------------------------

    def select(self) -> ScoredHint:
        """Select a hint using UCB1: score + c * sqrt(ln(total) / n_uses).

        Hints with 0 uses are prioritized (infinite UCB bonus).
        """
        if not self._hints:
            raise RuntimeError("HintPool is empty — provide seed_hints or call evolve() first.")

        # Prioritize unused hints
        unused = [h for h in self._hints if h.n_uses == 0]
        if unused:
            return unused[0]

        # UCB1
        ln_total = math.log(max(self._total_uses, 1))
        best_hint = None
        best_ucb = float("-inf")
        for h in self._hints:
            ucb = h.score + self.ucb_c * math.sqrt(ln_total / h.n_uses)
            if ucb > best_ucb:
                best_ucb = ucb
                best_hint = h
        return best_hint  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Score update
    # ------------------------------------------------------------------

    def update(self, hint: ScoredHint, improvement: float) -> None:
        """Update EMA score for a hint after observing teacher improvement."""
        hint.n_uses += 1
        self._total_uses += 1
        if hint.n_uses == 1:
            hint.score = improvement
        else:
            hint.score = (1 - self.ema_alpha) * hint.score + self.ema_alpha * improvement

        self._updates_since_save += 1
        if self._save_path is not None and self._autosave_every > 0 and self._updates_since_save >= self._autosave_every:
            self.save()

    # ------------------------------------------------------------------
    # Hard-solve collection (shared across all workflow instances)
    # ------------------------------------------------------------------

    def record_hard_solve(self, hard_solve: dict) -> None:
        """Record a rare correct solution on a hard problem.

        Thread-safe via the GIL (simple list append).  Called by workflow
        instances after each task.

        Args:
            hard_solve: Dict with keys ``"target"``, ``"nums"``,
                ``"response"`` (the correct solution), ``"student_pass_rate"``.
        """
        self._hard_solves.append(hard_solve)
        if len(self._hard_solves) > self._max_hard_solves:
            self._hard_solves = self._hard_solves[-self._max_hard_solves // 2 :]

    # ------------------------------------------------------------------
    # Evolution via LiteLLM
    # ------------------------------------------------------------------

    def should_evolve(self) -> bool:
        """Check whether the pool should evolve based on usage count.

        Uses ``_total_uses`` (incremented on every ``update()`` call across
        all workflow instances) instead of a per-instance step counter.
        """
        if self.evolve_every <= 0:
            return False
        return self._total_uses > 0 and self._total_uses - self._last_evolve_step >= self.evolve_every

    async def evolve(
        self,
        hard_solves: list[dict] | None = None,
    ) -> str | None:
        """Generate a new hint via an external LLM and replace the worst-performing one.

        Uses LiteLLM (async) to call the configured model (default: Gemini Flash
        via OpenRouter).  This is decoupled from the training rollout engine.

        Args:
            hard_solves: Optional override.  If None, uses the internally
                accumulated ``_hard_solves`` from :meth:`record_hard_solve`.

        Returns:
            The new hint text, or None if generation failed.
        """
        import litellm

        if self._evolving:
            logger.debug("[HintPool] Evolution already in progress, skipping")
            return None
        self._evolving = True

        self._last_evolve_step = self._total_uses
        if hard_solves is None:
            hard_solves = self._hard_solves[-5:]

        best = self.get_best(n=2)
        messages = _build_evolve_prompt(best, hard_solves)

        try:
            response = await litellm.acompletion(
                model=self.evolve_model,
                messages=messages,
                temperature=0.8,
                top_p=0.95,
                max_tokens=256,
            )
            text = response.choices[0].message.content or ""
            match = re.search(r"<hint>(.*?)</hint>", text, re.DOTALL | re.IGNORECASE)
            if not match:
                logger.warning(f"[HintPool] Evolution failed: no <hint> tags in response: {text[:200]}")
                return None

            new_text = match.group(1).strip()
            new_hint = ScoredHint(text=new_text, created_at=self._total_uses)

            # Replace worst hint if pool is full, otherwise append
            if len(self._hints) >= self.max_size:
                worst_idx = min(range(len(self._hints)), key=lambda i: self._hints[i].score)
                old = self._hints[worst_idx]
                logger.info(f"[HintPool] Replacing hint {worst_idx} (score={old.score:.3f}, uses={old.n_uses}) with new hint (total_uses={self._total_uses})")
                self._hints[worst_idx] = new_hint
            else:
                self._hints.append(new_hint)

            logger.info(f"[HintPool] Evolved new hint: {new_text[:200]}")
            return new_text

        except Exception:
            logger.exception("[HintPool] Evolution LLM call failed")
            return None
        finally:
            self._evolving = False

    def get_best(self, n: int = 2) -> list[ScoredHint]:
        """Return the top-N hints by score."""
        return sorted(self._hints, key=lambda h: h.score, reverse=True)[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        """Save the pool to a JSON file."""
        save_path = Path(path) if path is not None else self._save_path
        if save_path is None:
            return
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "hints": [h.to_dict() for h in self._hints],
            "total_uses": self._total_uses,
            "last_evolve_step": self._last_evolve_step,
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        self._updates_since_save = 0
        logger.info(f"[HintPool] Saved {len(self._hints)} hints → {save_path}")

    def load(self, path: str | Path | None = None) -> int:
        """Load the pool from a JSON file. Returns number of hints loaded."""
        load_path = Path(path) if path is not None else self._save_path
        if load_path is None or not load_path.exists():
            return 0
        with open(load_path) as f:
            data = json.load(f)
        self._hints = [ScoredHint.from_dict(d) for d in data["hints"]]
        self._total_uses = data.get("total_uses", 0)
        self._last_evolve_step = data.get("last_evolve_step", 0)
        logger.info(f"[HintPool] Loaded {len(self._hints)} hints ← {load_path}")
        return len(self._hints)


# ---------------------------------------------------------------------------
# Evolution prompt
# ---------------------------------------------------------------------------

_EVOLVE_SYSTEM = """
You are a strategy advisor for a number puzzle game called Countdown.

In Countdown, a player is given 3 numbers and a target. They must combine \
the numbers using +, -, *, / (each number used exactly once) to reach the target.

Your job: propose a NEW general-purpose strategy hint (at most 3 bullet \
points) that could help a solver do better. The hint should be broadly \
applicable, not specific to any one puzzle.
""".strip()


def _build_evolve_prompt(
    best_hints: list[ScoredHint],
    hard_solves: list[dict],
) -> list[dict]:
    """Build the LLM prompt for hint evolution.

    Args:
        best_hints: Top-scoring hints from the pool.
        hard_solves: "Lucky solves" — dicts with keys ``"target"``,
            ``"nums"``, ``"response"`` (the correct solution),
            ``"student_pass_rate"``.
    """
    parts: list[str] = []

    if best_hints:
        parts.append("[Best Strategies So Far]")
        for i, h in enumerate(best_hints, 1):
            parts.append(f"Strategy {i} (score={h.score:.3f}, used {h.n_uses} times):")
            parts.append(h.text)
            parts.append("")

    if hard_solves:
        parts.append("[Hard Problems With Rare Correct Solutions]")
        parts.append("These are problems the solver usually gets wrong, but occasionally solves correctly.")
        parts.append("Study the successful solutions to identify patterns that could become general strategies.")
        parts.append("")
        for ex in hard_solves[:5]:
            nums = ex.get("nums", [])
            target = ex.get("target", "?")
            rate = ex.get("student_pass_rate", 0.0)
            response = ex.get("response", "")
            parts.append(f"- Numbers: {', '.join(map(str, nums))}. Target: {target}. (pass rate: {rate:.0%})")
            if response:
                # Show just the answer portion, truncated
                parts.append(f"  Correct solution: {response[:300]}")
            parts.append("")

    parts.append(
        "Based on the patterns you see in the successful solutions above, "
        "propose a NEW strategy (different from existing ones) that might "
        "help the solver on similar hard problems. "
        "Output in <hint>...</hint> tags, at most 3 bullet points."
    )

    return [
        {"role": "system", "content": _EVOLVE_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ---------------------------------------------------------------------------
# Local test: real API call to evolve
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    SEED_HINTS = [
        ("- Try working backwards from the target number.\n- Look for pairs of numbers whose product or sum is close to the target.\n- Consider whether division can simplify the problem."),
        ("- Check if the target is achievable by combining just two numbers first.\n- Use the third number to adjust the result.\n- Subtraction and division are often overlooked."),
    ]

    async def test_evolve():
        pool = HintPool(
            max_size=5,
            seed_hints=SEED_HINTS,
            evolve_model=DEFAULT_EVOLVE_MODEL,
        )
        # Simulate some usage
        for h in pool._hints:
            h.score = 0.15
            h.n_uses = 10

        hard_solves = [
            {
                "target": 98,
                "nums": [44, 19, 35],
                "response": "44 + 19 + 35 = 98. All three numbers sum to the target.",
                "student_pass_rate": 0.2,
            },
            {
                "target": 87,
                "nums": [7, 12, 3],
                "response": "(12 - 3) * 7 = 63. Wait, that's wrong. 7 * 12 + 3 = 87. <answer>7 * 12 + 3</answer>",
                "student_pass_rate": 0.1,
            },
            {
                "target": 156,
                "nums": [25, 6, 31],
                "response": "25 * 6 + 31 = 181. No. (31 - 25) * 6 = 36. No. 25 * 6 = 150, 150 + 6 = 156 but can't reuse 6. 31 * 6 - 25 = 161. Hmm. (25 + 31) / 6... no. Actually (25 + 1) * 6 = 156, \
                    but I don't have 1. Let me try: 31 * (25 - 6*4)... <answer>(25 + 31) * 6 / (6/6)</answer>",
                "student_pass_rate": 0.05,
            },
        ]

        print("=== Pool before evolution ===")
        for i, h in enumerate(pool._hints):
            print(f"  [{i}] score={h.score:.3f} uses={h.n_uses}: {h.text[:80]}")

        # Record hard solves into the pool (as workflow instances would)
        for hs in hard_solves:
            pool.record_hard_solve(hs)

        print("\n=== Calling evolve (model={}) ===".format(pool.evolve_model))
        new_hint = await pool.evolve()

        print(f"\n=== New hint: {new_hint} ===")
        print("\n=== Pool after evolution ===")
        for i, h in enumerate(pool._hints):
            print(f"  [{i}] score={h.score:.3f} uses={h.n_uses}: {h.text[:80]}")

    asyncio.run(test_evolve())
