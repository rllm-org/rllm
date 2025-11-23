"""Decorators for trajectory tracking using session primitives."""

import asyncio
import inspect
from collections.abc import Callable
from functools import wraps

from rllm.sdk.protocol import TrajectoryView, trace_to_step_view
from rllm.sdk.shortcuts import session


def trajectory(name: str = "agent", **traj_metadata):
    """
    Decorator to mark a function as a trajectory.

    Creates a session internally and automatically converts each trace (LLM call)
    into a StepView. The decorator **changes the return value** - it returns
    TrajectoryView instead of the original return value.

    Each LLM call in the function becomes a step with a reward that can be set.

    Args:
        name: Name of the trajectory
        **traj_metadata: Additional metadata for the trajectory

    Returns:
        Decorator that wraps the function to return TrajectoryView

    Example:
        >>> @trajectory(name="solver")
        >>> async def solve_workflow(problem: str):
        ...     # Each LLM call becomes a step automatically
        ...     response1 = await llm.create(messages=[...])
        ...     response2 = await llm.create(messages=[...])
        ...     return "final_answer"

        >>> traj = await solve_workflow("What is 2+2?")
        >>> # Each trace is now a StepView
        >>> print(len(traj.steps))  # 2
        >>> # Set rewards on each step
        >>> traj.steps[0].reward = 1.0
        >>> traj.steps[1].reward = 0.5
        >>> # Set trajectory reward
        >>> traj.reward = sum(s.reward for s in traj.steps)
    """

    def decorator(func: Callable) -> Callable:
        # Get function signature for capturing args/kwargs
        sig = inspect.signature(func)

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> TrajectoryView:
                # Capture function arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_input = dict(bound_args.arguments)

                # Create a session for trajectory
                with session(trajectory_name=name, **traj_metadata) as traj_sess:
                    # Run the function
                    result = await func(*args, **kwargs)

                    # Convert each trace to a StepView. Prefer async accessor when available
                    # (e.g., OpenTelemetry sessions) to avoid async_to_sync recursion errors.
                    if hasattr(traj_sess, "llm_calls_async"):
                        traces = await traj_sess.llm_calls_async()  # type: ignore[attr-defined]
                    else:
                        traces = traj_sess.llm_calls
                    steps = [trace_to_step_view(trace) for trace in traces]

                    return TrajectoryView(
                        name=name,
                        steps=steps,
                        reward=0.0,  # Must be set manually by user
                        input=func_input,  # Function arguments
                        output=result,  # Function return value
                        metadata=traj_metadata if traj_metadata else None,
                    )

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> TrajectoryView:
                # Capture function arguments
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                func_input = dict(bound_args.arguments)

                # Create a session for trajectory
                with session(trajectory_name=name, **traj_metadata) as traj_sess:
                    # Run the function
                    result = func(*args, **kwargs)

                    # Convert each trace to a StepView
                    steps = [trace_to_step_view(trace) for trace in traj_sess.llm_calls]

                    return TrajectoryView(
                        name=name,
                        steps=steps,
                        reward=0.0,  # Must be set manually by user
                        input=func_input,  # Function arguments
                        output=result,  # Function return value
                        metadata=traj_metadata if traj_metadata else None,
                    )

            return sync_wrapper

    return decorator
