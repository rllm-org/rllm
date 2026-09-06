"""Harbor adapter for official CyberGym Level 1 task directories.

Lives under ``rllm.integrations.harbor`` because trials still run Harbor
compose. Not native CyberGym (``python3 -m cybergym.server`` on :8666)
and not ExploitGym / ACE.
"""

from rllm.integrations.harbor.cybergym.constants import DEFAULT_SCORING_MODE, SCORING_ANY_OF, SCORING_FINAL
from rllm.integrations.harbor.cybergym.detect import discover_cybergym_task_dirs, is_cybergym_task_dir
from rllm.integrations.harbor.cybergym.grade import dual_binary_reward, score_submissions
from rllm.integrations.harbor.cybergym.source import cybergym_task_to_row, load_cybergym_rows

__all__ = [
    "DEFAULT_SCORING_MODE",
    "SCORING_ANY_OF",
    "SCORING_FINAL",
    "cybergym_task_to_row",
    "discover_cybergym_task_dirs",
    "dual_binary_reward",
    "is_cybergym_task_dir",
    "load_cybergym_rows",
    "score_submissions",
]
