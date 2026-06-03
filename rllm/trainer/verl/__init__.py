# Lazy exports — avoid eagerly pulling in verl (DataProto, ray, etc.) at module import.
from .dataclass import AccumulatedData, ProcessedStepData

__all__ = [
    # backend (lazy)
    "VerlBackend",
    # data transformation (lazy — requires verl + torch)
    "transform_episodes_to_dataproto",
    "transform_trajectory_groups_to_dataproto",
    "update_dataproto_with_advantages",
    # dataclass
    "AccumulatedData",
    "ProcessedStepData",
]


def __getattr__(name):
    if name == "VerlBackend":
        from .verl_backend import VerlBackend as _VerlBackend

        return _VerlBackend
    if name in {
        "transform_episodes_to_dataproto",
        "transform_trajectory_groups_to_dataproto",
        "update_dataproto_with_advantages",
    }:
        from . import transform as _t

        return getattr(_t, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
