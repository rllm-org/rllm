__all__ = [
    "AccumulatedData",
    "ProcessedStepData",
    "compute_advantage_verl",
    "transform_episodes_to_dataproto",
    "transform_trajectory_groups_to_dataproto",
    "update_dataproto_with_advantages",
    "VerlBackend",
]


def __getattr__(name):
    if name in {"AccumulatedData", "ProcessedStepData"}:
        from .dataclass import AccumulatedData, ProcessedStepData

        exports = {
            "AccumulatedData": AccumulatedData,
            "ProcessedStepData": ProcessedStepData,
        }
        return exports[name]

    if name in {"transform_episodes_to_dataproto", "transform_trajectory_groups_to_dataproto", "update_dataproto_with_advantages"}:
        from .transform import transform_episodes_to_dataproto, transform_trajectory_groups_to_dataproto, update_dataproto_with_advantages

        exports = {
            "transform_episodes_to_dataproto": transform_episodes_to_dataproto,
            "transform_trajectory_groups_to_dataproto": transform_trajectory_groups_to_dataproto,
            "update_dataproto_with_advantages": update_dataproto_with_advantages,
        }
        return exports[name]

    if name == "compute_advantage_verl":
        from .verl_advantage import compute_advantage_verl

        return compute_advantage_verl

    if name == "VerlBackend":
        from .verl_backend import VerlBackend

        return VerlBackend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
