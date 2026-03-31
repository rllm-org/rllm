"""Serialization helpers for ExecutionResult."""

from __future__ import annotations

from rllm.sdk.sandbox.protocol import ExecutionResult


def serialize_execution_result(result: ExecutionResult) -> dict:
    """Convert an ExecutionResult to a JSON-serializable dict."""
    return {
        "success": result.success,
        "trajectories": result.trajectories,
        "session_uid": result.session_uid,
        "reward": result.reward,
        "error": result.error,
        "elapsed": result.elapsed,
    }


def deserialize_execution_result(data: dict) -> ExecutionResult:
    """Reconstruct an ExecutionResult from a dict."""
    return ExecutionResult(
        success=data["success"],
        trajectories=data.get("trajectories"),
        session_uid=data.get("session_uid", ""),
        reward=data.get("reward"),
        error=data.get("error"),
        elapsed=data.get("elapsed", 0.0),
    )
