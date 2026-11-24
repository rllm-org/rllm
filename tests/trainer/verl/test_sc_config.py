import pytest
from omegaconf import OmegaConf

from rllm.trainer.verl.sc_config import validate_score_centering_config


def config():
    return OmegaConf.create(
        {
            "rllm": {"algorithm": {"score_centering": {"enabled": True}, "rollout_correction": {}}},
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp"},
                "model": {},
                "rollout": {"name": "vllm", "engine_kwargs": {"vllm": {"max_logprobs": 32}}},
            },
        }
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor_rollout_ref.actor.strategy", "megatron"),
        ("actor_rollout_ref.rollout.name", "sglang"),
        ("actor_rollout_ref.rollout.temperature", 0.8),
        ("actor_rollout_ref.rollout.top_p", 0.9),
        ("actor_rollout_ref.rollout.top_k", 32),
        ("actor_rollout_ref.rollout.do_sample", False),
        ("actor_rollout_ref.rollout.dtype", "float16"),
        ("actor_rollout_ref.rollout.quantization", "fp8"),
        ("actor_rollout_ref.rollout.logprobs_mode", "raw_logprobs"),
        ("actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs", 20),
        ("actor_rollout_ref.actor.entropy_coeff", 0.01),
        ("actor_rollout_ref.actor.use_kl_loss", True),
        ("actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.model.use_fused_kernels", True),
        ("actor_rollout_ref.model.use_remove_padding", False),
        ("actor_rollout_ref.model.use_liger", True),
        ("actor_rollout_ref.model.mtp.enable", True),
        ("rllm.algorithm.loss_fn", "gspo"),
        ("rllm.algorithm.rollout_correction.tis_mode", "sequence"),
        ("rllm.algorithm.rollout_correction.bypass_mode", False),
        ("rllm.algorithm.rollout_correction.tis_cap", 0),
        ("rllm.algorithm.rollout_correction.tis_cap", float("nan")),
        ("rllm.algorithm.score_centering.chunk_size", 0),
        ("rllm.algorithm.score_centering.record_topk", True),
        ("rllm.async_training.partial_rollout", True),
        ("rllm.distill.enable", True),
    ],
)
def test_unsupported_combinations_fail_early(path, value):
    cfg = config()
    OmegaConf.update(cfg, path, value, force_add=True)
    with pytest.raises(ValueError):
        validate_score_centering_config(cfg)


def test_enabled_default_and_token_tis_with_bypass_are_valid():
    cfg = config()
    validate_score_centering_config(cfg)
    cfg.rllm.algorithm.rollout_correction.update({"bypass_mode": True, "tis_mode": "token", "tis_cap": 2.0})
    validate_score_centering_config(cfg)


def test_disabled_has_no_new_configuration_requirements():
    validate_score_centering_config(OmegaConf.create({"rllm": {"algorithm": {}}}))


def test_algorithm_config_parses_sc_settings_and_exports_them():
    from rllm.trainer.algorithms import AlgorithmConfig, ScoreCenteringConfig

    cfg = AlgorithmConfig.from_config(
        OmegaConf.create(
            {
                "adv_estimator": "grpo",
                "score_centering": {
                    "enabled": True,
                    "record_topk": 128,
                    "chunk_size": 16,
                },
            }
        )
    )
    assert cfg.score_centering == ScoreCenteringConfig(enabled=True, record_topk=128, chunk_size=16)


def test_multiple_epochs_rejected():
    cfg = config()
    OmegaConf.update(cfg, "actor_rollout_ref.actor.ppo_epochs", 2)
    with pytest.raises(ValueError, match="ppo_epochs"):
        validate_score_centering_config(cfg)


def test_multiple_minibatches_need_tis():
    cfg = config()
    OmegaConf.update(cfg, "actor_rollout_ref.actor.ppo_mini_batch_size", 8)
    OmegaConf.update(cfg, "data", {"train_batch_size": 32}, force_add=True)
    with pytest.raises(ValueError, match="one optimizer step"):
        validate_score_centering_config(cfg)
    OmegaConf.update(cfg, "rllm.algorithm.rollout_correction.tis_mode", "token")
    validate_score_centering_config(cfg)
    OmegaConf.update(cfg, "actor_rollout_ref.actor.ppo_mini_batch_size", 32)
    OmegaConf.update(cfg, "rllm.algorithm.rollout_correction.tis_mode", None)
    validate_score_centering_config(cfg)
