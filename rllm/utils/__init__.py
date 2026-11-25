"""Utilities for the rllm package."""

import sys

from rllm.utils.episode_logger import EpisodeLogger
from rllm.utils.source_extraction import extract_source_metadata
from rllm.utils.tracking import Tracking
from rllm.utils.visualization import VisualizationConfig, colorful_print, colorful_warning, visualize_trajectories

__all__ = ["EpisodeLogger", "compute_pass_at_k", "extract_source_metadata", "visualize_trajectories", "VisualizationConfig", "colorful_print", "colorful_warning", "Tracking"]


def __getattr__(name):
    if name == "compute_pass_at_k":
        from rllm.utils.compute_pass_at_k import compute_pass_at_k

        # Cache on the module: importing the submodule binds it as an attribute
        # of this package under the same name, which would otherwise shadow the
        # function on every later lookup.
        setattr(sys.modules[__name__], name, compute_pass_at_k)
        return compute_pass_at_k
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
