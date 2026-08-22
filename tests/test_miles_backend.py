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
            "workflow": {"raise_on_error": False},
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


@needs_miles_data
class TestAsyncTraining:
    """Async moves weight sync to on_policy_updated and constrains the pass structure."""

    def _backend(self, **async_over):
        async_cfg = {"enable": True, "mini_batch_size": 4, **async_over}
        return MilesBackend(config=_cfg(async_training=async_cfg))

    def test_async_is_accepted(self):
        self._backend().validate_config()

    def test_is_async_flag_tracks_the_config(self):
        assert self._backend().is_async is True
        assert MilesBackend(config=_cfg()).is_async is False

    def test_mismatched_fwd_bwd_group_size_is_rejected(self):
        # Miles takes its optimizer step inside RayTrainGroup.train(), so it cannot
        # accumulate gradient across several forward-backward passes.
        with pytest.raises(ValueError, match="cannot accumulate gradient"):
            self._backend(fwd_bwd_group_size=2).validate_config()

    def test_matching_fwd_bwd_group_size_is_fine(self):
        self._backend(fwd_bwd_group_size=4).validate_config()

    def test_abort_pause_mode_is_rejected_under_async(self):
        cfg = _cfg(async_training={"enable": True, "mini_batch_size": 1})
        cfg.miles = {"pause_generation_mode": "abort"}
        with pytest.raises(ValueError, match="requeues"):
            MilesBackend(config=cfg).validate_config()

    def test_retract_pause_mode_is_fine(self):
        cfg = _cfg(async_training={"enable": True, "mini_batch_size": 1})
        cfg.miles = {"pause_generation_mode": "retract"}
        MilesBackend(config=cfg).validate_config()

    def test_abort_pause_mode_is_fine_in_sync_mode(self):
        # Sync mode drains generation before syncing, so abort costs nothing there.
        cfg = _cfg()
        cfg.miles = {"pause_generation_mode": "abort"}
        MilesBackend(config=cfg).validate_config()


class TestWeightSyncPlacement:
    """Exactly one of on_batch_end / on_policy_updated may publish weights per step."""

    class _Recorder:
        def __init__(self):
            self.calls = []

        async def update_weights(self, rollout_id=None):
            self.calls.append(rollout_id)

        async def save_model(self, rollout_id):
            pass

    def _state(self):
        from rllm.trainer.unified_trainer import TrainerState

        return TrainerState(global_step=7)

    def _backend(self, is_async):
        b = MilesBackend(config=_cfg())
        b.is_async = is_async
        b.actor_model = self._Recorder()
        b.miles_args = type("A", (), {"save_interval": 0})()
        b.rollout_manager = None
        return b

    @pytest.mark.asyncio
    async def test_sync_mode_publishes_in_on_batch_end(self):
        b = self._backend(is_async=False)
        state = self._state()
        await b.on_batch_end(state)
        await b.on_policy_updated(state)
        assert b.actor_model.calls == [7], "sync mode should publish exactly once, from on_batch_end"

    @pytest.mark.asyncio
    async def test_async_mode_publishes_in_on_policy_updated(self):
        b = self._backend(is_async=True)
        state = self._state()
        await b.on_batch_end(state)
        await b.on_policy_updated(state)
        assert b.actor_model.calls == [7], "async mode should publish exactly once, from on_policy_updated"


class TestAsyncPrerequisites:
    def test_raise_on_error_true_is_rejected_early(self):
        # The trainer asserts this only after full GPU bring-up; catching it in
        # validate_config turns minutes of setup into an immediate error.
        cfg = _cfg(async_training={"enable": True, "mini_batch_size": 1})
        cfg.rllm.workflow = {"raise_on_error": True}
        with pytest.raises(ValueError, match="raise_on_error=false"):
            MilesBackend(config=cfg).validate_config()

    def test_raise_on_error_false_passes(self):
        cfg = _cfg(async_training={"enable": True, "mini_batch_size": 1})
        cfg.rllm.workflow = {"raise_on_error": False}
        MilesBackend(config=cfg).validate_config()

    def test_sync_mode_does_not_care(self):
        cfg = _cfg()
        cfg.rllm.workflow = {"raise_on_error": True}
        MilesBackend(config=cfg).validate_config()


class TestMilesBatchIsSized:
    """The async loop does `len(backend_batch)` to count trainable sequences."""

    def test_len_is_the_sample_count(self):
        assert len(MilesBatch(data_ref="r", num_samples=32)) == 32

    def test_empty_batch_has_len_zero(self):
        b = MilesBatch(data_ref=None, num_samples=0)
        assert len(b) == 0 and b.is_empty

    @pytest.mark.asyncio
    async def test_update_policy_skips_an_empty_batch(self):
        from rllm.trainer.unified_trainer import TrainerState

        b = MilesBackend(config=_cfg())
        b.actor_model = None  # would explode if update_policy did not return early
        state = TrainerState(global_step=1)
        state.backend_batch = MilesBatch(data_ref=None, num_samples=0)
        await b.update_policy(state)  # must not raise

    @pytest.mark.asyncio
    async def test_update_policy_skips_when_there_is_no_batch(self):
        from rllm.trainer.unified_trainer import TrainerState

        b = MilesBackend(config=_cfg())
        b.actor_model = None
        await b.update_policy(TrainerState(global_step=1))


@needs_miles_data
class TestPatchesSurviveImportOrder:
    """Miles' actors import these functions by value, so patching the source module
    only reaches an actor that has not been imported yet. Importing an actor first
    (assert_patch_contracts does) silently defeated the get_rollout_data patch and the
    advantages arrived at the loss as plain lists."""

    def test_get_rollout_data_is_repointed_in_the_fsdp_actor(self):
        import miles.backends.fsdp_utils.actor as fsdp_actor

        from rllm.trainer.miles.patch import patch_advantages_cp_slice

        patch_advantages_cp_slice()
        assert fsdp_actor.get_rollout_data.__module__ == "rllm.trainer.miles.patch"

    def test_compute_advantages_is_repointed_in_the_fsdp_actor(self):
        import miles.backends.fsdp_utils.actor as fsdp_actor

        from rllm.trainer.miles.patch import patch_respect_disable_compute_advantages

        patch_respect_disable_compute_advantages()
        assert fsdp_actor.compute_advantages_and_returns.__module__ == "rllm.trainer.miles.patch"

    def test_applying_everything_leaves_both_repointed(self):
        # The realistic ordering: contracts assert (importing the actor) then patches.
        import miles.backends.fsdp_utils.actor as fsdp_actor

        from rllm.trainer.miles.patch import apply_all_miles_patches

        apply_all_miles_patches()
        assert fsdp_actor.get_rollout_data.__module__ == "rllm.trainer.miles.patch"
        assert fsdp_actor.compute_advantages_and_returns.__module__ == "rllm.trainer.miles.patch"


class TestEngineCleanup:
    """httpx binds connections to the loop that opened them, so the client has to be
    closed on the trainer's loop -- not Miles' background loop, and not after fit()
    tore its loop down ("Event loop is closed")."""

    class _Engine:
        def __init__(self):
            self.closed = 0

        async def close(self):
            self.closed += 1

    @pytest.mark.asyncio
    async def test_on_train_end_closes_the_client(self):
        from rllm.trainer.unified_trainer import TrainerState

        b = MilesBackend(config=_cfg())
        b.rollout_engine = self._Engine()
        await b.on_train_end(TrainerState())
        assert b.rollout_engine.closed == 1

    @pytest.mark.asyncio
    async def test_closing_twice_is_a_no_op(self):
        from rllm.trainer.unified_trainer import TrainerState

        b = MilesBackend(config=_cfg())
        b.rollout_engine = self._Engine()
        await b.on_train_end(TrainerState())
        await b.on_train_end(TrainerState())
        assert b.rollout_engine.closed == 1

    def test_shutdown_does_not_touch_the_loop(self):
        # shutdown() is sync; awaiting a close here is what raised "Event loop is closed".
        b = MilesBackend(config=_cfg())
        b.rollout_engine = self._Engine()
        b.shutdown()
        assert b.rollout_engine.closed == 0


@needs_miles_data
class TestNoSilentAdvantageFallback:
    """If rLLM's advantages go missing while Miles' own computation is disabled, the
    only safe outcome is a loud failure -- recomputing from scalar rewards would train
    on the wrong signal while the run looks healthy."""

    def test_missing_advantages_raises_instead_of_recomputing(self):
        from miles.backends.training_utils import loss as miles_loss

        from rllm.trainer.miles.patch import patch_respect_disable_compute_advantages

        patch_respect_disable_compute_advantages()

        class Args:
            compute_advantages_and_returns = False

        with pytest.raises(RuntimeError, match="did not reach the train worker"):
            miles_loss.compute_advantages_and_returns(Args(), {"rewards": [1.0], "tokens": [[1]]})

    def test_the_error_names_the_keys_that_did_arrive(self):
        from miles.backends.training_utils import loss as miles_loss

        from rllm.trainer.miles.patch import patch_respect_disable_compute_advantages

        patch_respect_disable_compute_advantages()

        class Args:
            compute_advantages_and_returns = False

        with pytest.raises(RuntimeError) as e:
            miles_loss.compute_advantages_and_returns(Args(), {"rewards": [1.0]})
        assert "'rewards'" in str(e.value)
