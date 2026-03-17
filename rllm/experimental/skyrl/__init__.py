"""SkyRL backend for unified trainer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.experimental.rollout.skyrl_engine import SkyRLEngine
    from rllm.experimental.skyrl.skyrl_backend import SkyRLBackend

__all__ = [
    "SkyRLBackend",
    "SkyRLEngine",
]


def __getattr__(name: str):
    if name == "SkyRLBackend":
        from rllm.experimental.skyrl.skyrl_backend import SkyRLBackend as _SkyRLBackend

        return _SkyRLBackend
    if name == "SkyRLEngine":
        from rllm.experimental.rollout.skyrl_engine import SkyRLEngine as _SkyRLEngine

        return _SkyRLEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
