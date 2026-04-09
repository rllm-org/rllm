"""Optional utilities for GSD workflows.

* :class:`EmbeddingExperienceStore` — sentence-transformers-based memory of
  past ``(question, hint, outcome)`` tuples, used by the math workflow to
  condition hint generation on similar past problems.
* :class:`HintPool` — UCB1-scored pool of generic strategy hints with
  LiteLLM-based evolution, used by the countdown workflow.
* :class:`ScoringAccumulator` — shared asyncio batch accumulator for
  teacher-side scoring coroutines.  Optional — the new workflow uses
  ``compute_logprobs_async`` on a frozen client, which is cheap enough
  that per-task ``asyncio.gather`` is usually fine, but the accumulator
  is kept available for heavy-batch use cases.
"""

from rllm.experimental.gsd.utils.experience_store import EmbeddingExperienceStore
from rllm.experimental.gsd.utils.hint_pool import HintPool, ScoredHint
from rllm.experimental.gsd.utils.scoring_accumulator import ScoringAccumulator

__all__ = [
    "EmbeddingExperienceStore",
    "HintPool",
    "ScoredHint",
    "ScoringAccumulator",
]
