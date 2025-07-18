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
import os
import random
import re
import shutil
import tempfile
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed
from filelock import FileLock
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available, is_npu_available


class BaseCheckpointManager:
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_contents = checkpoint_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        raise NotImplementedError

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None):
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    @staticmethod
    def local_mkdir(path):
        if not os.path.isabs(path):
            working_dir = os.getcwd()
            path = os.path.join(working_dir, path)

        # Using hash value of path as lock file name to avoid long file name
        lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):  # Add timeout
                # make a new dir
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {path}: {e}")
            # Even if the lock is not acquired, try to create the directory
            os.makedirs(path, exist_ok=True)

        return path

    @staticmethod
    def get_rng_state():
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if is_cuda_available:
            rng_state["cuda"] = torch.cuda.get_rng_state()
        elif is_npu_available:
            rng_state["npu"] = torch.npu.get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if is_cuda_available:
            torch.cuda.set_rng_state(rng_state["cuda"])
        elif is_npu_available:
            torch.npu.set_rng_state(rng_state["npu"])


def is_valid_checkpoint(ckpt_path):
    """
    Returns True if the checkpoint directory has all required files/folders.
    Adjust the checks below to match your own project requirements.
    """

    # 1) Check for required top-level files
    required_pt_files = [
        'data.pt'
    ]
    for fname in required_pt_files:
        full_file = os.path.join(ckpt_path, fname)
        if not os.path.exists(full_file):
            print(f"Checkpoint {ckpt_path} is missing required file: {fname}")
            return False

    # 2) Check for 'actor' folder
    actor_dir = os.path.join(ckpt_path, "actor")
    if not os.path.isdir(actor_dir):
        print(f"Checkpoint {ckpt_path} is missing the 'actor' folder.")
        return False

    # 3) Check for subfolders in 'actor'
    checkpoint_dir = os.path.join(actor_dir, "checkpoint")
    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint {ckpt_path} is missing the 'checkpoint' directory inside 'actor'.")
        return False

    # hf_dir = os.path.join(actor_dir, "huggingface")
    # if not os.path.isdir(hf_dir):
    #     print(f"Checkpoint {ckpt_path} is missing the 'huggingface' directory inside 'actor'.")
    #     return False

    # 4) Check for .pt files in 'actor' that start with specific prefixes
    required_prefixes = [
        "model_world_size",
        "optim_world_size",
        "extra_state_world_size"
    ]
    actor_files = os.listdir(actor_dir)

    for prefix in required_prefixes:
        # Find files that start with the prefix and end with '.pt'
        matched_files = [f for f in actor_files if f.startswith(prefix) and f.endswith('.pt')]
        if not matched_files:
            print(f"Checkpoint {ckpt_path} is missing required .pt files with prefix '{prefix}' in the 'actor' folder.")
            return False

    # If all checks pass, the checkpoint is considered valid
    return True


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    # tracker_file = get_checkpoint_tracker_filename(path)
    # tracker_candidate = None
    # Try reading the tracker file first
    # if os.path.exists(tracker_file):
    #     try:
    #         with open(tracker_file, "rb") as f:
    #             iteration = int(f.read().decode())
    #         tracker_candidate = os.path.join(path, directory_format.format(iteration))
    #         if os.path.exists(tracker_candidate) and is_valid_checkpoint(tracker_candidate):
    #             print(f"Found valid checkpoint from tracker: {tracker_candidate}")
    #             return tracker_candidate
    #         else:
    #             print(f"Tracker checkpoint is invalid or missing: {tracker_candidate}")
    #     except Exception as e:
    #         print(f"Error reading tracker file {tracker_file}: {e}")
    # else:
    #     print(f"Checkpoint tracker file does not exist: {tracker_file}")

    # If tracker-based checkpoint is invalid, try all possible `global_step_X` folders
    try:
        entries = os.listdir(path)
    except Exception as e:
        print(f"Failed to list directory {path}: {e}")
        return None

    pattern = re.compile(r'^global_step_(\d+)$')
    candidates = []
    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            m = pattern.match(entry)
            if m:
                step = int(m.group(1))
                candidates.append((step, full_path))

    if not candidates:
        print(f"No checkpoint directories found in {path}")
        return None

    # Sort in descending order of global step
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Return the first valid checkpoint
    for step, candidate in candidates:
        if is_valid_checkpoint(candidate):
            print(f"Found valid checkpoint: {candidate} (global step: {step})")
            return candidate
        else:
            print(f"Checkpoint {candidate} is malformed, skipping.")

    print(f"No valid checkpoint found in {path}")
    return None





def get_checkpoint_tracker_filename(root_path: str):
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")
