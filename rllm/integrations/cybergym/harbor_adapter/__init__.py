"""Vendored official Harbor CyberGym adapter (Apache-2.0).

See ``NOTICE`` in this directory. Used by ``rllm.data.cybergym_builder``
to generate the 1,507 official Level 1 Harbor task directories from
``sunblaze-ucb/cybergym`` metadata. Heavy ``data/`` archives and runner
images are not checked in.
"""

from rllm.integrations.cybergym.harbor_adapter.adapter import (
    DIFFICULTY_LEVELS,
    SUBSET_TASK_IDS,
    TASK_TYPES,
    CyberGymRecord,
    CyberGymToHarbor,
)

__all__ = [
    "DIFFICULTY_LEVELS",
    "SUBSET_TASK_IDS",
    "TASK_TYPES",
    "CyberGymRecord",
    "CyberGymToHarbor",
]
