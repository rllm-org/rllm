# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from verl/verl/utils/tracking.py
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import numbers
import os
import pprint
import sys
import time
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any


def concat_dict_to_str(dict: dict, step):
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
        "ui",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None, source_metadata=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._finished = False  # Track whether finish() has been called

        # Nudge users to enable UI logging
        if "ui" not in default_backend:
            try:
                import os as _os

                from rllm.experimental.eval.config import load_ui_config

                ui_config = load_ui_config()
                has_key = bool(_os.getenv("RLLM_API_KEY") or ui_config.get("ui_api_key"))
                if has_key:
                    print("\033[1;34mrllm-ui\033[0m: Add \033[1m'ui'\033[0m to your logger list for live monitoring in rllm UI")
                else:
                    print("\033[1;34mrllm-ui\033[0m: Run \033[1mrllm login\033[0m to enable live monitoring in rllm UI")
            except Exception:
                pass

        rllm_config = config.get("rllm", {}) if config is not None else {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings = None
            if rllm_config and rllm_config.get("trainer", {}).get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=rllm_config["trainer"]["wandb_proxy"])
            wandb.init(project=project_name, name=experiment_name, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

        if "ui" in default_backend:
            self.logger["ui"] = UILogger(project_name, experiment_name, config, source_metadata=source_metadata)

    def log(self, data, step, backend=None, episodes=None, trajectory_groups=None):
        """Log metrics and optionally episodes/trajectory_groups to configured backends.

        Args:
            data: Dictionary of metrics to log
            step: Current training step
            backend: Optional list of backends to log to (default: all)
            episodes: Optional list of Episode objects (only used by UILogger)
            trajectory_groups: Optional list of TrajectoryGroup objects (only used by UILogger)
        """
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                if default_backend == "ui":
                    logger_instance.log(data=data, step=step, episodes=episodes, trajectory_groups=trajectory_groups)
                else:
                    logger_instance.log(data=data, step=step)

    def finish(self):
        """Explicitly finish and cleanup all loggers.

        This method should be called during controlled shutdown to ensure proper cleanup.
        It's safe to call multiple times - subsequent calls will be no-ops.
        """
        if self._finished:
            return

        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()
        if "ui" in self.logger:
            self.logger["ui"].finish()

        self.logger.clear()
        self._finished = True

    def __del__(self):
        """Destructor that ensures cleanup if finish() wasn't called explicitly.

        Note: Prefer calling finish() explicitly during shutdown rather than relying
        on __del__, as garbage collection timing can be unpredictable.
        """
        self.finish()


class TeeStream:
    """Wraps a stream to also send lines to the UI backend."""

    def __init__(self, original, client, session_id, stream_name="stdout"):
        self._original = original
        self._client = client
        self._session_id = session_id
        self._stream_name = stream_name
        self._line_buffer = ""
        self._log_buffer = []
        self._buffer_size = 20
        self._last_flush = time.time()
        self._flush_interval = 2.0

    def write(self, text):
        self._original.write(text)
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            if line.strip():
                self._log_buffer.append(line)
        # Auto-flush when buffer is full or interval elapsed
        if len(self._log_buffer) >= self._buffer_size or (self._log_buffer and time.time() - self._last_flush >= self._flush_interval):
            self._send_buffer()

    def flush(self):
        self._original.flush()
        # Flush remaining partial line
        if self._line_buffer.strip():
            self._log_buffer.append(self._line_buffer)
            self._line_buffer = ""
        if self._log_buffer:
            self._send_buffer()

    def isatty(self):
        # Report as TTY so libraries (Rich, tqdm, etc.) emit ANSI color codes
        return True

    def _send_buffer(self):
        if not self._log_buffer:
            return
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        logs = [{"session_id": self._session_id, "timestamp": now, "stream": self._stream_name, "message": line} for line in self._log_buffer]
        self._log_buffer = []
        self._last_flush = time.time()
        try:
            self._client.post("/api/logs/batch", json={"session_id": self._session_id, "logs": logs})
        except Exception:
            pass  # Silently ignore - don't break training if UI is down

    def __getattr__(self, name):
        return getattr(self._original, name)


class UILogger:
    """Logger that sends training data to the rLLM UI backend via HTTP.

    This logger sends both aggregated metrics and detailed episode data (including
    trajectories and step-by-step execution) to a FastAPI backend for visualization.

    Args:
        project_name: Name of the project
        experiment_name: Name of the experiment/run
        config: Training configuration dict
    """

    def __init__(self, project_name: str, experiment_name: str, config, source_metadata=None, session_type: str = "training"):
        import logging
        import queue
        import threading

        import httpx

        self.logger = logging.getLogger(__name__)
        self.session_type = session_type
        from rllm.experimental.eval.config import load_ui_config

        ui_config = load_ui_config()
        api_key = os.getenv("RLLM_API_KEY") or ui_config.get("ui_api_key")
        ui_url = os.getenv("RLLM_UI_URL")
        if not ui_url:
            ui_url = "https://ui.rllm-project.com" if api_key else "http://localhost:3000"
        self.ui_url = ui_url
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self.client = httpx.Client(base_url=self.ui_url, timeout=5.0, headers=headers)
        self._heartbeat_stop = threading.Event()

        try:
            # Create session with source metadata
            response = self.client.post(
                "/api/sessions",
                json={"project": project_name, "experiment": experiment_name, "config": config, "source_metadata": source_metadata or {}, "session_type": session_type},
            )
            response.raise_for_status()
            resp_data = response.json()
            self.session_id = resp_data.get("id") or resp_data.get("session_id")
            self.logger.info(f"UILogger initialized with session_id: {self.session_id}")

            # Build clickable session URL
            # For local dev, the frontend runs on a different port (default 5173)
            # than the API backend (default 3000). In production they share a domain.
            from urllib.parse import urlparse

            parsed = urlparse(self.ui_url)
            if parsed.hostname in ("localhost", "127.0.0.1"):
                frontend_base = f"{parsed.scheme}://localhost:5173"
            else:
                frontend_base = self.ui_url

            if session_type == "eval":
                self.session_url = f"{frontend_base}/evaluation/{self.session_id}"
            else:
                self.session_url = f"{frontend_base}/runs/{self.session_id}"

            # Print before TeeStream install so it goes to real stdout
            print(f"\033[1;34mrllm-ui\033[0m: View run at \033[4;34m{self.session_url}\033[0m")

            # Send initial heartbeat
            try:
                self.client.post(f"/api/sessions/{self.session_id}/heartbeat")
            except Exception:
                pass

            # Start heartbeat daemon thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            # Background log worker
            self._queue = queue.Queue(maxsize=64)
            self._worker_stop = threading.Event()
            self._worker_thread = threading.Thread(target=self._log_worker, daemon=True)
            self._worker_thread.start()

            # Install TeeStream to capture stdout/stderr
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = TeeStream(sys.stdout, self.client, self.session_id, "stdout")
            sys.stderr = TeeStream(sys.stderr, self.client, self.session_id, "stderr")
        except Exception as e:
            self.logger.warning(f"Failed to initialize UILogger: {e}")
            self.session_id = None

    def _heartbeat_loop(self):
        """Send heartbeat every 30 seconds until stopped."""
        while not self._heartbeat_stop.wait(30):
            if self.session_id is None:
                break
            try:
                self.client.post(f"/api/sessions/{self.session_id}/heartbeat")
            except Exception:
                pass

    def _log_worker(self):
        """Background worker that sends log data to the UI backend."""
        import queue

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._worker_stop.is_set():
                    return
                continue
            if item is None:  # sentinel for shutdown
                self._queue.task_done()
                return
            step, metrics_payload, episodes_payloads, groups_payloads = item
            try:
                self._send_log_data(step, metrics_payload, episodes_payloads, groups_payloads)
            except Exception as e:
                self.logger.warning(f"UILogger worker failed for step {step}: {e}")
            finally:
                self._queue.task_done()

    def _send_log_data(self, step, metrics_payload, episodes_payloads, groups_payloads):
        """Send pre-serialized data to UI backend. Runs on worker thread."""
        if metrics_payload is not None:
            try:
                self.logger.info(f"Sending metrics to UI [step {step}]")
                self.client.post("/api/metrics", json=metrics_payload)
            except Exception as e:
                self.logger.warning(f"Failed to send metrics to UI: {e}")

        if episodes_payloads:
            try:
                self.logger.info(f"Sending {len(episodes_payloads)} episodes to UI [step {step}]")
                # Batch upload — single request for all episodes
                batch_payload = {"session_id": self.session_id, "episodes": episodes_payloads}
                resp = self.client.post("/api/episodes/batch", json=batch_payload)
                resp.raise_for_status()
                self.logger.info(f"Sent {len(episodes_payloads)} episodes via batch endpoint")
            except Exception as batch_err:
                # Fallback to individual POSTs for older UI servers without batch endpoint
                self.logger.info(f"Batch endpoint failed ({batch_err}), falling back to individual POSTs")
                for ep in episodes_payloads:
                    try:
                        self.client.post("/api/episodes", json=ep)
                    except Exception as e:
                        self.logger.warning(f"Failed to send episode to UI: {e}")
                else:
                    self.logger.info(f"Sent {len(episodes_payloads)} episodes individually")

        if groups_payloads:
            try:
                self.logger.info(f"Sending {len(groups_payloads)} trajectory groups to UI [step {step}]")
                batch_payload = {"session_id": self.session_id, "groups": groups_payloads}
                resp = self.client.post("/api/trajectory-groups/batch", json=batch_payload)
                resp.raise_for_status()
                self.logger.info(f"Sent {len(groups_payloads)} trajectory groups via batch endpoint")
            except Exception as batch_err:
                self.logger.info(f"Trajectory groups batch endpoint failed ({batch_err}), falling back to individual POSTs")
                for gp in groups_payloads:
                    try:
                        self.client.post("/api/trajectory-groups", json=gp)
                    except Exception as e:
                        self.logger.warning(f"Failed to send trajectory group to UI: {e}")
                else:
                    self.logger.info(f"Sent {len(groups_payloads)} trajectory groups individually")

    def log(self, data, step, episodes=None, trajectory_groups=None):
        """Log metrics and optionally episodes/trajectory_groups.

        Serializes data immediately (to snapshot before mutation), then
        enqueues for background sending so the training loop is not blocked.
        """
        if self.session_id is None:
            return

        import json
        import queue

        # Serialize metrics (snapshot before mutation), skip empty data
        metrics_payload = None
        if data:
            try:
                metrics_json = json.loads(json.dumps(data, default=self._json_serializer))
                metrics_payload = {"session_id": self.session_id, "step": step, "data": metrics_json}
                self.logger.info(f"Queueing metrics to send to UI [step {step}]")
            except Exception as e:
                self.logger.warning(f"Failed to serialize metrics: {e}")

        # Serialize episodes
        episodes_payloads = None
        if episodes:
            try:
                _STEP_DROP_KEYS = {"prompt_ids", "response_ids", "logprobs", "model_output"}
                episodes_payloads = []
                for episode in episodes:
                    ep = episode.to_dict() if hasattr(episode, "to_dict") else episode.model_dump()
                    ep["session_id"] = self.session_id
                    ep["session_type"] = self.session_type
                    ep["step"] = step
                    ep["episode_id"] = ep.pop("id")
                    for traj in ep.get("trajectories", []):
                        for s in traj.get("steps", []):
                            for key in _STEP_DROP_KEYS:
                                s.pop(key, None)
                    episodes_payloads.append(json.loads(json.dumps(ep, default=self._json_serializer)))
                self.logger.info(f"Queueing {len(episodes_payloads)} episodes to send to UI [step {step}]")
            except Exception as e:
                self.logger.warning(f"Failed to serialize episodes: {e}")

        # Serialize trajectory groups
        groups_payloads = None
        if trajectory_groups:
            try:
                groups_payloads = []
                for group in trajectory_groups:
                    rewards = [float(t.reward) for t in group.trajectories if t.reward is not None]
                    gd = {
                        "session_id": self.session_id,
                        "step": step,
                        "group_id": group.group_id,
                        "num_trajectories": len(group.trajectories),
                        "avg_reward": sum(rewards) / len(rewards) if rewards else None,
                        "metadata": [{"episode_id": f"{m['task_id']}:{m['rollout_idx']}"} for m in group.metadata],
                    }
                    groups_payloads.append(json.loads(json.dumps(gd, default=self._json_serializer)))
                self.logger.info(f"Queueing {len(groups_payloads)} trajectory groups to send to UI [step {step}]")
            except Exception as e:
                self.logger.warning(f"Failed to serialize trajectory groups: {e}")

        # Enqueue for background sending
        try:
            self._queue.put((step, metrics_payload, episodes_payloads, groups_payloads), timeout=10)
        except queue.Full:
            self.logger.warning("UILogger queue full, dropping log for step %d", step)

    def log_eval_result(self, result) -> None:
        """Post an EvalResult to the UI backend.

        Args:
            result: An EvalResult dataclass from rllm.experimental.eval.results
        """
        if self.session_id is None:
            return

        try:
            payload = {
                "session_id": self.session_id,
                "dataset_name": result.dataset_name,
                "model": result.model,
                "agent": result.agent,
                "score": result.score,
                "total": result.total,
                "correct": result.correct,
                "errors": result.errors,
                "signal_averages": result.signal_averages,
                "items": [{"idx": item.idx, "reward": item.reward, "is_correct": item.is_correct, "error": item.error, "signals": item.signals} for item in result.items],
            }
            payload_json = json.loads(json.dumps(payload, default=self._json_serializer))
            response = self.client.post("/api/eval-results", json=payload_json)
            self.logger.info(f"Eval result posted, status: {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Failed to send eval result to UI: {e}")

    def _json_serializer(self, obj):
        """Convert numpy types and other non-JSON types to native Python."""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, "__dict__"):
            return str(obj)
        else:
            return str(obj)

    def finish(self, exit_code: int = 0):
        """Mark the session as complete, restore streams, and close HTTP client.

        Args:
            exit_code: Process exit code. 0 = completed, non-zero = failed.
        """
        if self.session_id is None:
            return

        # Drain the log queue and stop worker
        if hasattr(self, "_worker_thread"):
            self._worker_stop.set()
            self._queue.put(None)  # sentinel
            self._worker_thread.join(timeout=30)
            if self._worker_thread.is_alive():
                self.logger.warning("UILogger worker thread did not finish within 30s")

        # Stop heartbeat thread
        self._heartbeat_stop.set()

        try:
            # Flush and restore stdout/stderr
            if hasattr(self, "_original_stdout"):
                sys.stdout.flush()
                sys.stdout = self._original_stdout
            if hasattr(self, "_original_stderr"):
                sys.stderr.flush()
                sys.stderr = self._original_stderr
            status = "completed" if exit_code == 0 else "failed"
            self.client.post(
                f"/api/sessions/{self.session_id}/complete",
                json={"status": status, "exit_code": exit_code},
            )
        except Exception as e:
            self.logger.warning(f"Failed to complete session: {e}")
        finally:
            # Drain the log queue and stop worker
            if hasattr(self, "_worker_thread"):
                self._worker_stop.set()
                self._queue.put(None)  # sentinel
                self._worker_thread.join(timeout=30)
                if self._worker_thread.is_alive():
                    self.logger.warning("UILogger worker thread did not finish within 30s")
            self.client.close()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This invocation of ClearML logger\'s function is incorrect so this attribute was dropped. ')

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "w")

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(json.dumps(data) + "\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self):
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(r"[^/\w.\- :]")  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.

    def log(self, data, step):
        import mlflow

        def sanitize_key(key):
            # First replace @ with _at_ for backward compatibility
            sanitized = key.replace("@", "_at_")
            # Then replace any other invalid characters with _
            sanitized = self._invalid_chars_pattern.sub("_", sanitized)
            if sanitized != key:
                self.logger.warning("[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized)
            return sanitized

        results = {sanitize_key(k): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
