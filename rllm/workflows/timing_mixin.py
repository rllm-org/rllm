"""Timing tracking mixin for workflows to measure LLM, environment, and total execution time."""

import time
from datetime import datetime, timezone
from typing import Any


class TimingTrackingMixin:
    """Mixin to add timing tracking to workflows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timing_data = {
            "llm_time": 0.0,
            "env_time": 0.0,
            "reward_time": 0.0,
            "total_time": 0.0,
            "start_time": None,
            "end_time": None,
        }
        # Track per-step timing
        self._step_timings = []
        self._current_step_timing = None

    def start_timing(self):
        """Start timing the episode."""
        self._timing_data["start_time"] = time.time()  # Keep for calculation
        self._timing_data["start_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._timing_data["end_time"] = None
        self._timing_data["end_timestamp"] = None
        self._timing_data["llm_time"] = 0.0
        self._timing_data["env_time"] = 0.0
        self._timing_data["reward_time"] = 0.0
        self._step_timings = []
        self._current_step_timing = None

    def _start_new_step_timing(self):
        """Start timing for a new step."""
        if self._current_step_timing is not None:
            # Finish the previous step timing BEFORE appending
            self._finish_current_step_timing()
            self._step_timings.append(self._current_step_timing)

        self._current_step_timing = {
            "llm_time": 0.0,
            "env_time": 0.0,
            "step_start_time": time.time(),  # Keep for calculation
            "step_start_timestamp": datetime.now(timezone.utc).isoformat(),
            "step_end_time": None,
            "step_end_timestamp": None,
        }

    def _finish_current_step_timing(self):
        """Finish timing for the current step."""
        if self._current_step_timing is not None:
            self._current_step_timing["step_end_time"] = time.time()
            self._current_step_timing["step_end_timestamp"] = datetime.now(timezone.utc).isoformat()

    async def timed_llm_call(self, *args, **kwargs):
        """Wrapper for LLM calls with timing.

        Returns:
            ModelOutput from the rollout engine
        """
        # Start a new step timing when we make an LLM call
        self._start_new_step_timing()

        start = time.time()
        result = await self.rollout_engine.get_model_response(*args, **kwargs)
        duration = time.time() - start

        self._timing_data["llm_time"] += duration
        if self._current_step_timing is not None:
            self._current_step_timing["llm_time"] += duration

        return result

    async def timed_env_call(self, func, *args, **kwargs):
        """Wrapper for environment calls with timing.

        Args:
            func: The function to call (typically env.reset or env.step)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function call
        """
        start = time.time()
        result = await self.run_in_executor(func, *args, **kwargs)
        duration = time.time() - start

        self._timing_data["env_time"] += duration
        if self._current_step_timing is not None:
            self._current_step_timing["env_time"] += duration

        return result

    def add_reward_time(self, duration: float):
        """Add reward computation time.

        Args:
            duration: Time spent computing rewards
        """
        self._timing_data["reward_time"] += duration

    def finalize_timing(self):
        """Finalize timing calculations.

        Returns:
            Dictionary with timing information including start_timestamp and end_timestamp as ISO 8601 strings
        """
        # Finish the last step timing if it exists
        if self._current_step_timing is not None:
            self._finish_current_step_timing()
            self._step_timings.append(self._current_step_timing)
            self._current_step_timing = None

        # Set end time for the episode
        if self._timing_data["start_time"] is not None:
            self._timing_data["end_time"] = time.time()
            self._timing_data["end_timestamp"] = datetime.now(timezone.utc).isoformat()
            self._timing_data["total_time"] = self._timing_data["end_time"] - self._timing_data["start_time"]

        return {
            "start_timestamp": self._timing_data["start_timestamp"],  # ISO 8601 timestamp string
            "end_timestamp": self._timing_data["end_timestamp"],  # ISO 8601 timestamp string
            "llm_time": self._timing_data["llm_time"],
            "env_time": self._timing_data["env_time"],
            "reward_time": self._timing_data["reward_time"],
            "total_time": self._timing_data["total_time"],
        }

    def postprocess_episode(self, episode, termination_reason=None, error=None):
        """Override to add timing metrics to episode.

        This should be called from the subclass's postprocess_episode method.
        """
        # Get timing data before calling parent
        timing_metrics = self.finalize_timing()

        # Call parent's postprocess if it exists
        if hasattr(super(), "postprocess_episode"):
            episode = super().postprocess_episode(episode, termination_reason, error)

        # Add timing to episode info
        episode.info["timing"] = timing_metrics

        # Add per-trajectory metrics
        for trajectory in episode.trajectories:
            trajectory.info["num_steps"] = len(trajectory.steps)

            # Calculate trajectory-level timing (start and end timestamps)
            trajectory_start = None
            trajectory_end = None

            if trajectory.steps:
                # Get start time from first step
                if self._step_timings:
                    trajectory_start = self._step_timings[0].get("step_start_timestamp")
                    trajectory_end = self._step_timings[-1].get("step_end_timestamp")
                elif timing_metrics.get("start_timestamp") is not None:
                    # Fallback to episode timing if no step timings available
                    trajectory_start = timing_metrics["start_timestamp"]
                    trajectory_end = timing_metrics["end_timestamp"]

            # Add trajectory-level timing with start and end timestamps as ISO 8601 strings
            trajectory.info["timing"] = {
                "start_timestamp": trajectory_start,  # ISO 8601 timestamp string
                "end_timestamp": trajectory_end,  # ISO 8601 timestamp string
                "llm_time": timing_metrics["llm_time"],
                "env_time": timing_metrics["env_time"],
                "reward_time": timing_metrics["reward_time"],
                "total_time": timing_metrics["total_time"],
            }

            # Add per-step timing to each step (with ISO 8601 timestamps)
            for i, step in enumerate(trajectory.steps):
                if i < len(self._step_timings):
                    step_timing = self._step_timings[i]
                    step.info["timing"] = {
                        "start_timestamp": step_timing.get("step_start_timestamp"),  # ISO 8601 timestamp string
                        "end_timestamp": step_timing.get("step_end_timestamp"),  # ISO 8601 timestamp string
                        "llm_time": step_timing.get("llm_time", 0.0),
                        "env_time": step_timing.get("env_time", 0.0),
                    }
                else:
                    # No timing data available for this step
                    step.info["timing"] = {
                        "start_timestamp": None,
                        "end_timestamp": None,
                        "llm_time": 0.0,
                        "env_time": 0.0,
                    }

        return episode

    def reset(self, task: Any = None, uid: str | None = None):
        """Override reset to start timing.

        Subclasses should call this if they override reset.
        """
        # Start timing when resetting
        self.start_timing()

        # Call parent's reset if it exists
        if hasattr(super(), "reset"):
            return super().reset(task, uid)
