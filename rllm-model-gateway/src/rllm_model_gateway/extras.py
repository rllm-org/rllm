"""Pinned keys for NormalizedResponse.extras.

Names mirror ``rllm.experimental.rollout.ModelOutput`` and
``rllm.agents.agent.Step`` so traces round-trip cleanly into the engine.
"""

PROMPT_IDS = "prompt_ids"  # list[int]
COMPLETION_IDS = "completion_ids"  # list[int]
LOGPROBS = "logprobs"  # list[float] — completion-token logprobs
PROMPT_LOGPROBS = "prompt_logprobs"  # list[float] — prompt-token logprobs
ROUTING_MATRICES = "routing_matrices"  # list[str] — per-completion-token serialized
