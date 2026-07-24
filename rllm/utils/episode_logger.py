"""Episode JSON Logger for saving detailed episode information."""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from rllm.types import Episode


class EpisodeLogger:
    """Logger to save episodes to individual JSON files with step and data hash."""

    def __init__(self, base_dir: str, subdirectory: str = "episodes", flat_layout: bool = False):
        """Initialize the episode logger.

        Args:
            base_dir: Base directory for episode logs. Can be configured via
                     config.trainer.episode_log_dir
                     (default: "logs/${trainer.project_name}/${trainer.experiment_name}")
            subdirectory: Subdirectory within base_dir for episodes (default: "episodes")
                         Final path will be: {base_dir}/{subdirectory}/
            flat_layout: Write episodes directly into the subdirectory instead
                         of creating step/epoch directories.
        """
        self.log_dir = Path(base_dir) / subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flat_layout = flat_layout

    @staticmethod
    def compute_task_hash(task: Any, length: int = 8) -> str:
        """Compute a hash from the task data.

        Args:
            task: The task dictionary or data
            length: Length of the hash to use (default 8 chars)

        Returns:
            Hash string
        """
        # Convert task to a stable string representation
        task_str = json.dumps(task, sort_keys=True, default=str)
        # Compute SHA256 hash
        hash_obj = hashlib.sha256(task_str.encode("utf-8"))
        # Return first `length` characters of hex digest
        return hash_obj.hexdigest()[:length]

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the directory path for a specific training or validation step.

        Args:
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0

        Returns:
            Path object for the step directory
        """
        if self.flat_layout:
            return self.log_dir
        step_dir = self.log_dir / f"{mode}_step_{step}_epoch_{epoch}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def get_episode_filename(self, episode: Episode, step: int) -> str:
        """Generate filename for an episode.

        Format: episode_hash{task_hash}_id{episode_id}.json

        Args:
            episode: The episode to save
            step: Current training step (not used in filename, but kept for compatibility)

        Returns:
            Filename string
        """
        task_hash = self.compute_task_hash(episode.task)
        # Clean episode_id to make it filesystem-safe
        episode_id_safe = str(episode.id).replace(":", "_").replace("/", "_")

        filename = f"episode_hash{task_hash}_id{episode_id_safe}.json"
        return filename

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0):
        """Log a single episode to its own JSON file in a step-specific directory.

        Args:
            episode: The episode to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        episode_data = {
            "training_step": step,
            "epoch": epoch,
            "episode_id": episode.id,
            "session_id": episode.session_id,
            "task": episode.task,
            "task_hash": self.compute_task_hash(episode.task),
            "is_correct": episode.is_correct,
            "termination_reason": (episode.termination_reason.value if episode.termination_reason else None),
            "metrics": episode.metrics,
            "metadata": episode.metadata,
            "timing": episode.info.get("timing", {}),
            "trajectories": [],
        }

        for traj in episode.trajectories:
            traj_data = {
                "name": traj.name,
                "uid": traj.uid,
                "reward": traj.reward,
                "num_steps": len(traj.steps),
                "timing": traj.info.get("timing", {}),
                "steps": [
                    {
                        "observation": step.observation,
                        "thought": step.thought,
                        "action": step.action,
                        "reward": step.reward,
                        "done": step.done,
                        "model_response": step.model_response,
                        "model_output": (
                            step.model_output.to_dict()
                            if step.model_output is not None and hasattr(step.model_output, "to_dict")
                            else step.model_output
                        ),
                        "chat_completions": step.chat_completions,
                        "timing": step.info.get("timing", {}),  # Add step-level timing
                    }
                    for step in traj.steps
                ],
            }
            episode_data["trajectories"].append(traj_data)

        # Write to individual file in step-specific directory
        step_dir = self.get_step_dir(step, mode, epoch)
        filename = self.get_episode_filename(episode, step)
        filepath = step_dir / filename

        try:
            with open(filepath, "w") as f:
                json_str = json.dumps(episode_data, indent=2, default=str)
                f.write(json_str + "\n")
                f.flush()  # Ensure data is written to disk
        except Exception as e:
            print(f"Error writing episode to {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise

    def log_episodes(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0):
        """Log multiple episodes, each to its own file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        print(f"[EpisodeLogger] Logging {len(episodes)} episodes for step={step}, mode={mode}, epoch={epoch}")
        for i, episode in enumerate(episodes):
            try:
                self.log_episode(episode, step, mode, epoch)
                print(f"[EpisodeLogger] Successfully logged episode {i + 1}/{len(episodes)}: {episode.id}")
            except Exception as e:
                print(f"[EpisodeLogger] Failed to log episode {i + 1}/{len(episodes)}: {e}")
                raise

    def log_episodes_batch(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0, batch_summary: bool = True):
        """Log multiple episodes and optionally create a batch summary in step-specific directory.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
            batch_summary: Whether to create a summary file for the batch
        """
        # Log individual episodes
        self.log_episodes(episodes, step, mode, epoch)

        # Optionally create batch summary in step-specific directory
        if batch_summary and episodes:
            summary_data = {
                "training_step": step,
                "epoch": epoch,
                "mode": mode,
                "num_episodes": len(episodes),
                "episode_files": [self.get_episode_filename(ep, step) for ep in episodes],
                "summary_stats": {
                    "total_correct": sum(1 for ep in episodes if ep.is_correct),
                    "total_incorrect": sum(1 for ep in episodes if not ep.is_correct),
                    "accuracy": sum(1 for ep in episodes if ep.is_correct) / len(episodes) if episodes else 0,
                    "avg_trajectories_per_episode": sum(len(ep.trajectories) for ep in episodes) / len(episodes) if episodes else 0,
                },
            }

            step_dir = self.get_step_dir(step, mode, epoch)
            summary_file = step_dir / "batch_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)


class BackendBatchLogger:
    """Persist exact pre-forward backend datums as compressed JSON."""

    def __init__(self, base_dir: str, subdirectory: str = "backend_batches"):
        self.log_dir = Path(base_dir) / subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _tensor_data_to_dict(value: Any) -> dict[str, Any]:
        dtype = getattr(value, "dtype", None)
        return {
            "data": list(value.data),
            "dtype": getattr(dtype, "value", str(dtype)),
            "shape": value.shape,
            "sparse_crow_indices": value.sparse_crow_indices,
            "sparse_col_indices": value.sparse_col_indices,
        }

    @classmethod
    def _loss_inputs_to_dict(cls, datum: Any) -> dict[str, Any]:
        return {
            name: cls._tensor_data_to_dict(value)
            for name, value in datum.loss_fn_inputs.items()
        }

    @classmethod
    def _datum_to_dict(cls, source: Any, submitted: Any) -> dict[str, Any]:
        return {
            "model_input": submitted.model_input.model_dump(mode="json"),
            "source_loss_fn_inputs": cls._loss_inputs_to_dict(source),
            "submitted_loss_fn_inputs": cls._loss_inputs_to_dict(submitted),
        }

    def log_batch(
        self,
        source_datums: list[Any],
        submitted_datums: list[Any],
        step: int,
        forward_backward_index: int,
        operation: str,
    ) -> Path:
        """Write full source data and the exact datums submitted to one training call."""
        import zstandard

        if len(source_datums) != len(submitted_datums):
            raise ValueError(
                f"source/submitted datum count mismatch: {len(source_datums)} != {len(submitted_datums)}"
            )
        payload = {
            "training_step": step,
            "forward_backward_index": forward_backward_index,
            "operation": operation,
            "num_datums": len(source_datums),
            "datums": [
                self._datum_to_dict(source, submitted)
                for source, submitted in zip(source_datums, submitted_datums, strict=True)
            ],
        }
        path = self.log_dir / f"step_{step:06d}_forward_backward_{forward_backward_index:03d}.json.zst"
        with self._lock, open(path, "wb") as raw:
            with zstandard.ZstdCompressor(level=3).stream_writer(raw) as compressed:
                compressed.write(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"))
        return path
