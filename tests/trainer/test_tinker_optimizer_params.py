import asyncio
from types import SimpleNamespace

import tinker
from omegaconf import OmegaConf

import rllm.trainer.tinker.tinker_policy_trainer as policy_trainer_module
from rllm.trainer.algorithms import AlgorithmConfig
from rllm.trainer.tinker.tinker_backend import TinkerBackend
from rllm.trainer.tinker.tinker_policy_trainer import TinkerPolicyTrainer


class _CompletedFuture:
    def __init__(self, result=None):
        self.result = result

    async def result_async(self):
        return self.result


class _CapturingTrainingClient:
    def __init__(self):
        self.adam_params = None
        self.optim_step_response = tinker.types.OptimStepResponse(metrics={"grad_norm": 3.2, "adam:step": 7.0})

    async def optim_step_async(self, adam_params):
        self.adam_params = adam_params
        return _CompletedFuture(self.optim_step_response)


def _make_backend(monkeypatch, *, fused):
    monkeypatch.setattr(tinker, "ServiceClient", lambda **_kwargs: object())
    config = OmegaConf.create(
        {
            "tinker_base_url": None,
            "fuse_forward_backward_and_optim_step": fused,
            "training": {
                "learning_rate": 2e-5,
                "beta1": 0.8,
                "beta2": 0.9,
                "eps": 1e-7,
                "weight_decay": 0.123,
                "grad_clip_norm": 4.5,
            },
        }
    )
    return TinkerBackend(config)


def _make_policy_trainer():
    trainer = object.__new__(TinkerPolicyTrainer)
    trainer.training_client = _CapturingTrainingClient()
    trainer.algorithm_config = AlgorithmConfig()
    return trainer


def test_non_fused_backend_passes_configured_optimizer_params_to_adam(monkeypatch):
    backend = _make_backend(monkeypatch, fused=False)
    backend.policy_trainer = _make_policy_trainer()
    trainer_state = SimpleNamespace(global_step=1, total_steps=10, timing_dict={}, extra_info={}, metrics={})

    asyncio.run(backend.update_policy(trainer_state))

    assert backend.policy_trainer.training_client.adam_params.weight_decay == 0.123
    assert backend.policy_trainer.training_client.adam_params.grad_clip_norm == 4.5
    assert trainer_state.metrics["train/grad_norm"] == 3.2
    assert trainer_state.metrics["train/adam/step"] == 7.0


def test_fused_backend_passes_configured_optimizer_params_to_adam(monkeypatch):
    backend = _make_backend(monkeypatch, fused=True)
    backend.policy_trainer = _make_policy_trainer()
    backend.policy_trainer._get_vocab_size = lambda: 1

    async def _get_forward_backward_futures(**_kwargs):
        result = SimpleNamespace(loss_fn_outputs=[], metrics={})
        return [asyncio.sleep(0, result=result)]

    backend.policy_trainer._get_forward_backward_futures = _get_forward_backward_futures
    monkeypatch.setattr(policy_trainer_module, "transform_trajectory_groups_to_datums", lambda *_args, **_kwargs: ([], {}))
    trainer_state = SimpleNamespace(
        global_step=1,
        total_steps=10,
        trajectory_groups=[object()],
        timing_dict={},
        extra_info={},
        metrics={},
    )

    asyncio.run(backend.process_backend_batch(trainer_state))

    assert backend.policy_trainer.training_client.adam_params.weight_decay == 0.123
    assert backend.policy_trainer.training_client.adam_params.grad_clip_norm == 4.5
    assert trainer_state.metrics["train/grad_norm"] == 3.2
    assert trainer_state.metrics["train/adam/step"] == 7.0
