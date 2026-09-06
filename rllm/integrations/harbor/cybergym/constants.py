"""Harbor CyberGym Level 1 constants.

Timeouts match official Harbor CyberGym / native ``--timeout 1200``.
Do not use Harbor's 60s wait-for-main as a task timeout, and do not use
the 5400s cybergym-e2e (find-PoC-and-patch) budget.
"""

from __future__ import annotations

from typing import Literal

# Dual-binary exclusions: timeout / SIGKILL (bash 137 and Python -9).
EXCLUDE_EXIT_CODES: frozenset[int] = frozenset({124, 137, -9})

AGENT_TIMEOUT_SEC = 1200.0
VERIFIER_TIMEOUT_SEC = 180.0
POC_EXEC_TIMEOUT_SEC = 60.0
IMAGE_BUILD_TIMEOUT_SEC = 1800.0

# Training/session cap: agent + verifier + slack. Image pulls use task.toml.
DEFAULT_SESSION_TIMEOUT_SEC = 1800.0

SCORING_ANY_OF = "any_of"
SCORING_FINAL = "final_submission"
ScoringMode = Literal["any_of", "final_submission"]
DEFAULT_SCORING_MODE: ScoringMode = SCORING_ANY_OF

LEVEL1 = 1

TASK_SERVER_HOST = "task-server"
TASK_SERVER_PORT = 9111
