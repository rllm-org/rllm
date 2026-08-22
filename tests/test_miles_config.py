"""Miles config bridge: OmegaConf -> argv rendering."""

import pytest
from omegaconf import OmegaConf

from rllm.trainer.miles.miles_config import (
    PINNED_FLAGS,
    build_block,
    render_argv,
    validate_pinned,
)

# Stand-in for the arity map miles_arity() introspects off Miles' own parser.
ARITY = {
    "colocate": 0,
    "disable_rollout_global_dataset": 0,
    "disable_compute_advantages_and_returns": 0,
    "hf_checkpoint": 1,
    "num_rollout": 1,
    "rollout_num_gpus": 1,
    "rollout_function_path": 1,
    "eval_function_path": 1,
    "eval_prompt_data": "+",
    "ft_components": "+",
}


class TestRenderArgv:
    def test_scalar_becomes_flag_and_value(self):
        assert render_argv({"hf_checkpoint": "Qwen/Qwen3-4B"}, ARITY) == ["--hf-checkpoint", "Qwen/Qwen3-4B"]

    def test_snake_case_key_becomes_kebab_flag(self):
        assert render_argv({"rollout_num_gpus": 8}, ARITY) == ["--rollout-num-gpus", "8"]

    def test_true_zero_arity_flag_is_emitted_bare(self):
        assert render_argv({"disable_rollout_global_dataset": True}, ARITY) == ["--disable-rollout-global-dataset"]

    def test_false_zero_arity_flag_is_omitted(self):
        assert render_argv({"colocate": False}, ARITY) == []

    def test_none_is_omitted_so_miles_default_stands(self):
        assert render_argv({"num_rollout": None, "hf_checkpoint": "m"}, ARITY) == ["--hf-checkpoint", "m"]

    def test_nargs_flag_takes_every_value(self):
        assert render_argv({"ft_components": ["rollout", "train"]}, ARITY) == ["--ft-components", "rollout", "train"]

    def test_empty_nargs_list_is_omitted(self):
        assert render_argv({"ft_components": []}, ARITY) == []

    def test_list_for_scalar_flag_is_rejected(self):
        with pytest.raises(TypeError, match="single value"):
            render_argv({"hf_checkpoint": ["a", "b"]}, ARITY)

    def test_scalar_for_nargs_flag_is_rejected(self):
        with pytest.raises(TypeError, match="takes a list"):
            render_argv({"ft_components": "rollout"}, ARITY)

    def test_unknown_flag_infers_arity_from_type(self):
        # Megatron's half of the CLI is not in the introspected map.
        out = render_argv({"tensor_model_parallel_size": 4, "sequence_parallel": True, "moe_layer_freq": [1, 0]}, ARITY)
        assert out == ["--tensor-model-parallel-size", "4", "--sequence-parallel", "--moe-layer-freq", "1", "0"]

    def test_no_arity_map_still_renders(self):
        assert render_argv({"hf_checkpoint": "m", "colocate": False}) == ["--hf-checkpoint", "m"]


class TestValidatePinned:
    def test_overriding_a_pinned_flag_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be overridden"):
            validate_pinned({"rollout_function_path": "my.custom.rollout"})

    def test_error_names_every_clash(self):
        with pytest.raises(ValueError) as e:
            validate_pinned({"rollout_function_path": "x", "disable_rollout_global_dataset": False})
        assert "miles.rollout_function_path" in str(e.value)
        assert "miles.disable_rollout_global_dataset" in str(e.value)

    def test_colocate_is_rejected(self):
        with pytest.raises(ValueError, match="colocate"):
            validate_pinned({"colocate": True})

    def test_colocate_false_is_fine(self):
        validate_pinned({"colocate": False})

    def test_num_epoch_is_rejected_because_miles_asserts_its_own_dataset(self):
        with pytest.raises(ValueError, match="total_epochs"):
            validate_pinned({"num_epoch": 3})

    def test_unrelated_flags_pass(self):
        validate_pinned({"rollout_num_gpus": 8, "train_backend": "fsdp"})


class TestBuildBlock:
    def _cfg(self, **miles):
        return OmegaConf.create(
            {
                "model": {"name": "Qwen/Qwen3-4B"},
                "rllm": {
                    "data": {"train_batch_size": 32, "max_prompt_length": 4096},
                    "trainer": {"save_freq": 20, "test_freq": 5},
                },
                "miles": miles,
            }
        )

    def test_pinned_flags_are_always_present(self):
        block, _ = build_block(self._cfg())
        for flag, value in PINNED_FLAGS.items():
            assert block[flag] == value

    def test_shared_keys_are_mirrored_from_rllm_namespace(self):
        block, _ = build_block(self._cfg())
        assert block["hf_checkpoint"] == "Qwen/Qwen3-4B"
        assert block["rollout_batch_size"] == 32
        assert block["rollout_max_prompt_len"] == 4096
        assert block["save_interval"] == 20

    def test_eval_interval_is_never_mirrored(self):
        # miles_validate_args: "Evaluation datasets must be configured when
        # eval_interval is set." rLLM runs validation itself, so Miles must not eval.
        block, _ = build_block(self._cfg())
        assert "eval_interval" not in block

    def test_explicit_miles_value_wins_over_shared_key(self):
        block, _ = build_block(self._cfg(hf_checkpoint="/local/ckpt"))
        assert block["hf_checkpoint"] == "/local/ckpt"

    def test_total_steps_sets_num_rollout(self):
        block, _ = build_block(self._cfg(), total_steps=250)
        assert block["num_rollout"] == 250

    def test_extra_args_are_split_out_not_rendered_as_a_flag(self):
        block, extra = build_block(self._cfg(extra_args=["--moe-grouped-gemm"]))
        assert "extra_args" not in block
        assert extra == ["--moe-grouped-gemm"]

    def test_missing_shared_source_is_simply_absent(self):
        cfg = OmegaConf.create({"rllm": {}, "miles": {"train_backend": "fsdp"}})
        block, _ = build_block(cfg)
        assert "hf_checkpoint" not in block
        assert block["train_backend"] == "fsdp"

    def test_round_trip_to_argv(self):
        block, extra = build_block(self._cfg(rollout_num_gpus=8, train_backend="fsdp"), total_steps=100)
        argv = render_argv(block, ARITY) + extra
        assert "--disable-rollout-global-dataset" in argv
        assert "--disable-compute-advantages-and-returns" in argv
        assert argv[argv.index("--rollout-function-path") + 1] == "miles.rollout.sleep_rollout.sleep"
        assert argv[argv.index("--num-rollout") + 1] == "100"


class TestShippedBackendYaml:
    """Guards rllm/trainer/config/rllm/backend/miles.yaml against drift."""

    def _load(self):
        from pathlib import Path

        import rllm.trainer.config as cfg_pkg

        path = Path(cfg_pkg.__file__).parent / "rllm" / "backend" / "miles.yaml"
        cfg = OmegaConf.load(path)
        OmegaConf.update(cfg, "training.group_size", 8)  # ??? in the shipped file
        OmegaConf.update(cfg, "rllm.data.train_batch_size", 16)
        OmegaConf.update(cfg, "rllm.trainer.project_name", "rllm-miles-test")
        OmegaConf.update(cfg, "rllm.trainer.experiment_name", "t")
        return cfg

    def test_yaml_does_not_set_any_pinned_flag(self):
        # A pinned flag creeping into the shipped yaml would make every run fail.
        cfg = self._load()
        build_block(cfg)  # raises if the yaml collides with PINNED_FLAGS

    def test_yaml_renders_to_argv(self):
        cfg = self._load()
        block, extra = build_block(cfg, total_steps=50)
        argv = render_argv(block, ARITY) + extra
        assert argv[argv.index("--train-backend") + 1] == "fsdp"
        assert argv[argv.index("--num-rollout") + 1] == "50"
        assert "--disable-rollout-global-dataset" in argv
        # null entries (ref_load, tensor_model_parallel_size) stay out of argv
        assert "--ref-load" not in argv
        assert "--tensor-model-parallel-size" not in argv
        # empty extra_args must not become a flag
        assert "--extra-args" not in argv

    def test_yaml_declares_the_miles_backend(self):
        assert self._load().rllm.backend == "miles"


class TestRejectedGenerationFlags:
    """Miles flags that configure its own generation path, which rLLM replaces."""

    def test_fully_async_is_rejected_because_it_overrides_the_rollout_pin(self):
        from rllm.trainer.miles.miles_config import REJECTED_GENERATION_FLAGS

        assert "fully_async" in REJECTED_GENERATION_FLAGS
        with pytest.raises(ValueError, match="fully_async"):
            validate_pinned({"fully_async": True})

    def test_multi_lora_is_rejected(self):
        with pytest.raises(ValueError, match="rollout function of its own"):
            validate_pinned({"multi_lora": True})

    def test_session_server_is_rejected(self):
        with pytest.raises(ValueError, match="rLLM owns tokenization"):
            validate_pinned({"use_session_server": "v2"})

    def test_positive_spelling_of_the_dataset_flag_is_caught(self):
        # The negated flag is the real one; setting the dest name is a silent no-op otherwise.
        with pytest.raises(ValueError, match="disable-rollout-global-dataset"):
            validate_pinned({"rollout_global_dataset": True})

    def test_falsy_values_are_not_rejected(self):
        validate_pinned({"fully_async": False, "multi_lora": False, "partial_rollout": False})

    def test_error_explains_why_not_just_that(self):
        with pytest.raises(ValueError) as e:
            validate_pinned({"fully_async": True})
        assert "sleep_rollout" in str(e.value)


class TestMilesTrainIters:
    """Miles derives its LR-schedule length from these three flags:

        train_iters = num_rollout * rollout_batch_size * n_samples_per_prompt // global_batch_size

    Zero makes the FSDP scheduler assert; too small silently shortens the schedule.
    """

    def _cfg(self, *, async_enable=False, mini_batch=4, train_batch=16, n=8):
        return OmegaConf.create(
            {
                "model": {"name": "m"},
                "rllm": {
                    "data": {"train_batch_size": 1 if async_enable else train_batch},
                    "rollout": {"n": n},
                    "trainer": {"save_freq": 20},
                    "async_training": {"enable": async_enable, "mini_batch_size": mini_batch},
                },
                "miles": {"global_batch_size": mini_batch * n if async_enable else train_batch * n},
            }
        )

    def _train_iters(self, block, num_rollout):
        return num_rollout * block["rollout_batch_size"] * block["n_samples_per_prompt"] // block["global_batch_size"]

    def test_group_size_is_mirrored(self):
        block, _ = build_block(self._cfg())
        assert block["n_samples_per_prompt"] == 8

    def test_sync_train_iters_equals_the_step_count(self):
        block, _ = build_block(self._cfg(), total_steps=60)
        assert self._train_iters(block, 60) == 60

    def test_async_uses_mini_batch_size_not_the_pinned_train_batch_size(self):
        # The trainer forces rllm.data.train_batch_size to 1 under async.
        block, _ = build_block(self._cfg(async_enable=True), total_steps=20)
        assert block["rollout_batch_size"] == 4
        assert self._train_iters(block, 20) == 20

    def test_async_train_iters_is_never_zero(self):
        # The regression that broke the first async run: rollout_batch_size=1 and
        # n_samples_per_prompt=1 gave 20 * 1 * 1 // 32 == 0.
        block, _ = build_block(self._cfg(async_enable=True), total_steps=20)
        assert self._train_iters(block, 20) > 0


class TestRolloutCorrection:
    """Async trains on data from older weight versions; without TIS the ratio is taken
    against a fresh pi_old, the update is biased, and the policy can degrade. This was
    unmapped, so rllm.algorithm.rollout_correction.* was silently ignored."""

    def _cfg(self, correction=None, async_on=True, staleness=0.5):
        return OmegaConf.create(
            {
                "model": {"name": "m"},
                "rllm": {
                    "data": {"train_batch_size": 1},
                    "rollout": {"n": 8},
                    "trainer": {"save_freq": 20},
                    "async_training": {"enable": async_on, "mini_batch_size": 32, "staleness_threshold": staleness},
                    "algorithm": {"rollout_correction": correction or {}},
                },
                "miles": {},
            }
        )

    def test_token_tis_enables_miles_tis(self):
        block, _ = build_block(self._cfg({"tis_mode": "token", "tis_cap": 2.0}))
        assert block["use_tis"] is True
        assert block["tis_clip"] == 2.0

    def test_tis_cap_is_forwarded(self):
        block, _ = build_block(self._cfg({"tis_mode": "token", "tis_cap": 1.5}))
        assert block["tis_clip"] == 1.5

    def test_no_tis_mode_leaves_the_flags_alone(self):
        block, _ = build_block(self._cfg({"tis_mode": None}))
        assert "use_tis" not in block

    def test_bypass_mode_uses_rollout_logprobs(self):
        block, _ = build_block(self._cfg({"bypass_mode": True}))
        assert block["use_rollout_logprobs"] is True

    def test_bypass_false_leaves_miles_recomputing_pi_old(self):
        block, _ = build_block(self._cfg({"bypass_mode": False}))
        assert "use_rollout_logprobs" not in block

    def test_sequence_tis_is_rejected_not_silently_downgraded(self):
        with pytest.raises(ValueError, match="token-level"):
            build_block(self._cfg({"tis_mode": "sequence"}))

    def test_explicit_miles_flag_wins(self):
        cfg = self._cfg({"tis_mode": "token", "tis_cap": 2.0})
        cfg.miles = {"tis_clip": 3.0}
        block, _ = build_block(cfg)
        assert block["tis_clip"] == 3.0

    def test_stale_async_without_correction_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            build_block(self._cfg({}, staleness=1.0))
        assert "biased by the policy lag" in caplog.text

    def test_on_policy_async_does_not_warn(self, caplog):
        build_block(self._cfg({}, staleness=0.0))
        assert "biased by the policy lag" not in caplog.text
