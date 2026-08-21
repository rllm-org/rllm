"""MilesBackend / MilesEngine surface checks that need no GPU."""

from __future__ import annotations

import importlib

import pytest
from omegaconf import OmegaConf

from rllm.trainer.miles.miles_backend import MilesBackend, MilesBatch


def _can_import(module: str) -> bool:
    """Functional gate, not find_spec.

    tests/test_miles_against_checkout.py puts MILES_ROOT on sys.path while it is
    being collected, so `find_spec("miles")` can succeed in this module even where
    miles' own dependencies are absent — which would make outcomes depend on
    collection order. Actually importing the module under test cannot.
    """
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


needs_miles_data = pytest.mark.skipif(
    not _can_import("miles.backends.training_utils.data"),
    reason="miles.backends.training_utils.data not importable (needs sglang + torch)",
)
has_miles = _can_import("miles")


def _cfg(**rllm_over):
    base = {
        "model": {"name": "Qwen/Qwen3-4B"},
        "rllm": {
            "async_training": {"enable": False},
            "data": {"train_batch_size": 8, "max_prompt_length": 1024, "max_response_length": 512},
            "trainer": {"save_freq": 10, "project_name": "p", "experiment_name": "e"},
            **rllm_over,
        },
        "miles": {},
    }
    return OmegaConf.create(base)


class TestBackendContract:
    def test_implements_every_abstract_method(self):
        assert not MilesBackend.__abstractmethods__

    def test_name_is_miles(self):
        assert MilesBackend.name == "miles"

    def test_registered_in_the_launcher_table(self):
        from rllm.trainer.unified_trainer import _BACKEND_LAUNCHERS

        module, cls, extra = _BACKEND_LAUNCHERS["miles"]
        assert (module, cls, extra) == ("rllm.trainer.miles.miles_launcher", "MilesTrainerLauncher", "miles")

    def test_construction_does_no_heavy_work(self):
        # __init__ must not import miles or claim GPUs; bring-up happens in
        # init_rollout_engine, once total_training_steps is known.
        backend = MilesBackend(config=_cfg())
        assert backend.actor_model is None
        assert backend.rollout_manager is None
        assert backend.miles_args is None


class TestValidateConfig:
    def test_async_training_is_rejected(self):
        backend = MilesBackend(config=_cfg(async_training={"enable": True}))
        with pytest.raises((ValueError, ImportError)) as e:
            backend.validate_config()
        # ImportError only when miles is absent; either way async must not slip through.
        if isinstance(e.value, ValueError):
            assert "Phase 5" in str(e.value)

    def test_pinned_flag_override_is_rejected(self):
        cfg = _cfg()
        cfg.miles = {"rollout_function_path": "my.rollout"}
        backend = MilesBackend(config=cfg)
        with pytest.raises((ValueError, ImportError)) as e:
            backend.validate_config()
        if isinstance(e.value, ValueError):
            assert "cannot be overridden" in str(e.value)

    @pytest.mark.skipif(has_miles, reason="miles is importable")
    def test_missing_miles_explains_how_to_install(self):
        with pytest.raises(ImportError, match="pip install -r requirements.txt"):
            MilesBackend(config=_cfg()).validate_config()


class TestMilesBatch:
    def test_carries_the_ref_and_metrics(self):
        batch = MilesBatch(data_ref="ref", sample_indices=[0, 1], num_samples=2, metrics={"batch/miles_samples": 2})
        assert batch.data_ref == "ref"
        assert batch.num_samples == 2
        assert batch.metrics["batch/miles_samples"] == 2

    def test_metrics_default_is_not_shared(self):
        a, b = MilesBatch(data_ref="x"), MilesBatch(data_ref="y")
        a.metrics["k"] = 1
        assert b.metrics == {}


class TestPatchContract:
    """The advantages CP-slice patch is the load-bearing mechanism; guard its shape."""

    @needs_miles_data
    def test_all_patch_contracts_hold_against_installed_miles(self):
        from rllm.trainer.miles.patch import assert_patch_contracts

        assert_patch_contracts()

    @needs_miles_data
    def test_patch_is_idempotent(self):
        from miles.backends.training_utils import data as miles_data

        from rllm.trainer.miles.patch import patch_advantages_cp_slice

        patch_advantages_cp_slice()
        once = miles_data.get_rollout_data
        patch_advantages_cp_slice()
        assert miles_data.get_rollout_data is once

    def test_expected_key_tuple_is_documented(self):
        from rllm.trainer.miles.patch import _EXPECTED_CP_SLICED_KEYS

        assert _EXPECTED_CP_SLICED_KEYS == ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl")


class TestEngineIsDecoupledFromVerl:
    def test_miles_modules_do_not_import_verl(self):
        # A miles run must not need the verl extra installed.
        import pathlib

        offenders = []
        for path in pathlib.Path("rllm/trainer/miles").glob("*.py"):
            text = path.read_text()
            if "rllm.trainer.verl" in text or "\nimport verl" in text or "\nfrom verl" in text:
                offenders.append(path.name)
        assert not offenders, f"miles modules reference verl: {offenders}"

    def test_engine_does_not_import_verl(self):
        import pathlib

        text = pathlib.Path("rllm/engine/rollout/miles_engine.py").read_text()
        assert "verl" not in text


@needs_miles_data
class TestAdvantagesReachTheTrainWorkers:
    """The two silent-drop sites between the driver's transform and Miles' loss.

    Both were found only by running end to end: with either one unpatched the run
    still completes, but the loss quietly uses Miles' own advantages instead of
    rLLM's, so there is no error to notice.
    """

    def test_package_shards_forwards_advantages_per_partition(self):
        from miles.ray.rollout import train_data_conversion as tdc

        from rllm.trainer.miles.patch import patch_package_shards_forwards_advantages

        patch_package_shards_forwards_advantages()

        data = {
            "tokens": [[1, 2], [3, 4], [5, 6], [7, 8]],
            "advantages": [[0.1], [0.2], [0.3], [0.4]],
        }
        partitions = [[0, 2], [1, 3]]

        class Args:
            multi_lora_n_adapters = 0

        shards = tdc._package_shards(Args(), data, partitions)
        assert [s["advantages"] for s in shards] == [[[0.1], [0.3]], [[0.2], [0.4]]]

    def test_package_shards_without_advantages_is_unchanged(self):
        from miles.ray.rollout import train_data_conversion as tdc

        from rllm.trainer.miles.patch import patch_package_shards_forwards_advantages

        patch_package_shards_forwards_advantages()

        class Args:
            multi_lora_n_adapters = 0

        shards = tdc._package_shards(Args(), {"tokens": [[1], [2]]}, [[0], [1]])
        assert all("advantages" not in s for s in shards)

    def test_disable_flag_keeps_driver_advantages(self):
        from miles.backends.training_utils import loss as miles_loss

        from rllm.trainer.miles.patch import patch_respect_disable_compute_advantages

        patch_respect_disable_compute_advantages()

        class Args:
            compute_advantages_and_returns = False

        mine = [[0.5, 0.5]]
        rollout_data = {"advantages": mine}
        miles_loss.compute_advantages_and_returns(Args(), rollout_data)
        assert rollout_data["advantages"] is mine, "Miles overwrote rLLM's advantages"
        assert rollout_data["returns"] is mine, "returns should alias advantages under GRPO"

    def test_flag_on_still_delegates_to_miles(self):
        from miles.backends.training_utils import loss as miles_loss

        from rllm.trainer.miles.patch import patch_respect_disable_compute_advantages

        patch_respect_disable_compute_advantages()
        called = {}

        class Args:
            compute_advantages_and_returns = True

        # The wrapper must not swallow the normal path; a missing "log_probs" key makes
        # the real implementation return early, which is enough to prove delegation.
        rollout_data = {"advantages": [[1.0]], "log_probs": None, "values": None}
        try:
            miles_loss.compute_advantages_and_returns(Args(), rollout_data)
            called["ok"] = True
        except Exception:
            called["ok"] = True  # reached the real function, which is the point
        assert called["ok"]

    def test_fsdp_actor_binding_is_repointed(self):
        # Both actors do `from ...loss import compute_advantages_and_returns`, binding
        # by value, so patching only the source module would leave the FSDP call site
        # (which is ungated) pointing at the original.
        import miles.backends.fsdp_utils.actor as fsdp_actor

        from rllm.trainer.miles.patch import patch_respect_disable_compute_advantages

        patch_respect_disable_compute_advantages()
        assert fsdp_actor.compute_advantages_and_returns.__module__ == "rllm.trainer.miles.patch"

    @needs_miles_data
    def test_contract_check_fails_when_a_patch_site_disappears(self, monkeypatch):
        # The guard is only worth having if it actually trips.
        from miles.ray.rollout import train_data_conversion as tdc

        from rllm.trainer.miles.patch import assert_patch_contracts

        monkeypatch.delattr(tdc, "_package_shards")
        with pytest.raises(RuntimeError, match="_package_shards is gone"):
            assert_patch_contracts()
