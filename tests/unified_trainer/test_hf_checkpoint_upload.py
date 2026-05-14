"""Tests for optional Hugging Face checkpoint upload."""

import sys
import types

from omegaconf import OmegaConf


def test_upload_checkpoint_to_hf(monkeypatch, tmp_path):
    from rllm.experimental.verl import utils as utils_mod

    calls = {}

    class FakeHfApi:
        def upload_folder(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))

    cfg = OmegaConf.create(
        {
            "trainer": {
                "hf_upload": True,
                "hf_repo_id": "org/repo",
                "experiment_name": "exp",
            }
        }
    )

    utils_mod._upload_checkpoint_to_hf(cfg, str(tmp_path), 10)

    assert calls == {
        "repo_id": "org/repo",
        "folder_path": str(tmp_path),
        "path_in_repo": "checkpoints/exp/global_step_10",
        "commit_message": "Upload checkpoint global_step_10",
    }
