"""Harbor CyberGym Level 1: thin Harbor runner over official task directories.

Catalog is the official 1,507 tasks from ``sunblaze-ucb/cybergym``.
Not native CyberGym (``python3 -m cybergym.server`` on :8666) and not
ExploitGym / ACE. See ``docs/cybergym/RLLM-HARBOR-PROTOCOL.md``.
"""

from rllm.integrations.cybergym.constants import DEFAULT_SCORING_MODE, SCORING_ANY_OF, SCORING_FINAL
from rllm.integrations.cybergym.detect import discover_cybergym_task_dirs, is_cybergym_task_dir
from rllm.integrations.cybergym.evaluator import CyberGymEvaluator
from rllm.integrations.cybergym.grade import dual_binary_reward, score_submissions
from rllm.integrations.cybergym.runtime import CyberGymRuntime
from rllm.integrations.cybergym.source import cybergym_task_to_row, load_cybergym_rows

__all__ = [
    "DEFAULT_SCORING_MODE",
    "SCORING_ANY_OF",
    "SCORING_FINAL",
    "CyberGymEvaluator",
    "CyberGymRuntime",
    "cybergym_task_to_row",
    "discover_cybergym_task_dirs",
    "dual_binary_reward",
    "is_cybergym_task_dir",
    "load_cybergym_rows",
    "score_submissions",
]
