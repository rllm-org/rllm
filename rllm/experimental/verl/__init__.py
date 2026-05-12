"""Lazy exports for the experimental Verl integration.

Keep lightweight dataclasses importable without requiring the optional `verl`
dependency to be installed. Heavy modules are loaded on first attribute access.
"""

from __future__ import annotations

from rllm.experimental.verl.dataclass import AccumulatedData, ProcessedStepData

__all__ = [
    # data transformation
    "transform_episodes_to_dataproto",
    "transform_trajectory_groups_to_dataproto",
    "update_dataproto_with_advantages",
    # backend
    "VerlBackend",
    "VerlDPOBackend",
    # dataclass
    "AccumulatedData",
    "ProcessedStepData",
]


def __getattr__(name: str):
    if name in {
        "transform_episodes_to_dataproto",
        "transform_trajectory_groups_to_dataproto",
        "update_dataproto_with_advantages",
    }:
        from rllm.experimental.verl.transform import (
            transform_episodes_to_dataproto,
            transform_trajectory_groups_to_dataproto,
            update_dataproto_with_advantages,
        )

        return {
            "transform_episodes_to_dataproto": transform_episodes_to_dataproto,
            "transform_trajectory_groups_to_dataproto": transform_trajectory_groups_to_dataproto,
            "update_dataproto_with_advantages": update_dataproto_with_advantages,
        }[name]

    if name == "VerlBackend":
        from rllm.experimental.verl.verl_backend import VerlBackend

        return VerlBackend

    if name == "VerlDPOBackend":
        from rllm.experimental.verl.dpo_backend import VerlDPOBackend

        return VerlDPOBackend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
