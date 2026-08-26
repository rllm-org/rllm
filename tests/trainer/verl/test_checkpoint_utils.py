import os

import pytest
from omegaconf import OmegaConf

from rllm.trainer.verl.utils import load_checkpoint, save_checkpoint


class _RecordingWorker:
    def __init__(self, tracker_path):
        self.tracker_path = tracker_path
        self.tracker_value_during_save = None
        self.call = None
        self.load_call = None

    def save_checkpoint(self, local_path, remote_path, global_step, max_ckpt_to_keep=None):
        if os.path.exists(self.tracker_path):
            self.tracker_value_during_save = self.tracker_path.read_text()
        self.call = (local_path, remote_path, global_step, max_ckpt_to_keep)
        os.makedirs(local_path, exist_ok=True)

    def load_checkpoint(self, local_path, del_local_after_load=False):
        self.load_call = (local_path, del_local_after_load)


def _config(tmp_path, *, strategy, async_save):
    return OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "resume_mode": "auto",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
            },
            "actor_rollout_ref": {
                "actor": {
                    "strategy": strategy,
                    "checkpoint": {"async_save": async_save},
                }
            },
        }
    )


@pytest.mark.parametrize(
    ("strategy", "async_save"),
    [
        ("fsdp", False),
        ("fsdp", True),
        ("fsdp2", True),
        ("megatron", False),
    ],
)
def test_save_checkpoint_writes_tracker_after_synchronous_save(tmp_path, strategy, async_save):
    tracker_path = tmp_path / "latest_checkpointed_iteration.txt"
    tracker_path.write_text("4")
    worker = _RecordingWorker(tracker_path)

    save_checkpoint(_config(tmp_path, strategy=strategy, async_save=async_save), 5, worker)

    assert worker.tracker_value_during_save == "4"
    assert tracker_path.read_text() == "5"
    assert worker.call == (str(tmp_path / "global_step_5" / "actor"), None, 5, None)


def test_save_checkpoint_leaves_async_megatron_tracker_to_finalize_callback(tmp_path):
    tracker_path = tmp_path / "latest_checkpointed_iteration.txt"
    tracker_path.write_text("4")
    worker = _RecordingWorker(tracker_path)

    save_checkpoint(_config(tmp_path, strategy="megatron", async_save=True), 5, worker)

    assert worker.tracker_value_during_save == "4"
    assert tracker_path.read_text() == "4"


def test_async_configured_fsdp_checkpoint_is_discoverable_for_auto_resume(tmp_path):
    config = _config(tmp_path, strategy="fsdp", async_save=True)
    worker = _RecordingWorker(tmp_path / "latest_checkpointed_iteration.txt")

    save_checkpoint(config, 5, worker)
    resumed_step = load_checkpoint(config, worker)

    assert resumed_step == 5
    assert worker.load_call == (str(tmp_path / "global_step_5" / "actor"), False)
