"""Tests for the auxiliary-loss framework (rllm.trainer.algorithms.aux_loss).

Covers the spec/registry, config resolution (incl. ECHO back-compat sugar), the
managed-backend datum builder, and the in-process (verl) executor math via a
faithful reimplementation of ``seq-mean-token-mean`` aggregation (verl's
``agg_loss`` is not importable without the verl extra).

See design/auxiliary-losses.md.
"""

import math

import pytest
from omegaconf import OmegaConf

from rllm.trainer.algorithms import AlgorithmConfig
from rllm.trainer.algorithms.aux_loss import (
    AUX_LOSS_REGISTRY,
    MASK_ACTION,
    MASK_OBSERVATION,
    AuxiliaryLoss,
    EnvPredictionLoss,
    build_aux_losses,
    get_aux_loss,
    register_aux_loss,
)


def _algo(**kw):
    base = {"adv_estimator": "grpo", "norm_adv_by_std_in_grpo": True}
    base.update(kw)
    return AlgorithmConfig.from_config(OmegaConf.create(base))


# --- registry ---------------------------------------------------------------


def test_env_prediction_registered():
    assert "env_prediction" in AUX_LOSS_REGISTRY
    assert get_aux_loss("env_prediction") is EnvPredictionLoss
    assert EnvPredictionLoss.mask == MASK_OBSERVATION


def test_get_aux_loss_unknown_raises():
    with pytest.raises(ValueError):
        get_aux_loss("does_not_exist")


# --- spec validation --------------------------------------------------------


def test_unknown_mask_raises():
    @register_aux_loss("_bad_mask_test")
    class _Bad(AuxiliaryLoss):
        mask = "not_a_mask"

    with pytest.raises(ValueError):
        _Bad(coef=0.1)
    del AUX_LOSS_REGISTRY["_bad_mask_test"]


def test_requires_not_implemented_raises():
    @register_aux_loss("_needs_forward_test")
    class _NeedsForward(AuxiliaryLoss):
        mask = MASK_ACTION
        requires = ("teacher",)

    with pytest.raises(NotImplementedError):
        _NeedsForward(coef=0.1)
    del AUX_LOSS_REGISTRY["_needs_forward_test"]


def test_default_weight_is_uniform():
    assert EnvPredictionLoss(coef=0.05).weight(ctx=None) is None


# --- build_aux_losses / config ---------------------------------------------


def test_echo_sugar_yields_env_prediction():
    """adv_estimator=echo (env_loss_coef auto 0.05) builds one env_prediction loss."""
    losses = build_aux_losses(_algo(adv_estimator="echo"))
    assert [(loss.name, loss.coef, loss.mask) for loss in losses] == [("env_prediction", 0.05, MASK_OBSERVATION)]


def test_grpo_builds_no_aux_losses():
    assert build_aux_losses(_algo()) == []


def test_explicit_aux_losses_list():
    losses = build_aux_losses(_algo(aux_losses=[{"type": "env_prediction", "coef": 0.02}]))
    assert len(losses) == 1 and losses[0].coef == 0.02


def test_explicit_list_takes_precedence_over_sugar():
    """An explicit env_prediction entry is not double-added by the env_loss_coef sugar."""
    losses = build_aux_losses(_algo(adv_estimator="echo", aux_losses=[{"type": "env_prediction", "coef": 0.03}]))
    assert len(losses) == 1 and losses[0].coef == 0.03


def test_env_loss_coef_on_grpo_enables_env_prediction():
    losses = build_aux_losses(_algo(env_loss_coef=0.05))
    assert len(losses) == 1 and losses[0].name == "env_prediction" and losses[0].coef == 0.05


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        build_aux_losses(_algo(aux_losses=[{"type": "nope", "coef": 0.1}]))


def test_missing_coef_raises():
    with pytest.raises(ValueError):
        build_aux_losses(_algo(aux_losses=[{"type": "env_prediction"}]))


# --- managed-backend datum builder (tinker / fireworks) ---------------------


def test_managed_aux_positions_and_weights():
    tinker = pytest.importorskip("tinker")
    from tinker.types.tensor_data import TensorData

    from rllm.trainer.tinker.aux_loss import aux_positions, build_aux_ce_datum

    mask = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # actions: 0,1,5 ; observations: 2,3,4
    echo = EnvPredictionLoss(coef=0.05)
    obs_pos = aux_positions(echo, mask)
    assert obs_pos == [False, False, True, True, True, False]

    targets = TensorData(data=[10, 11, 12, 13, 14, 15], dtype="int64")
    datum = build_aux_ce_datum(tinker.ModelInput.from_ints([10, 11, 12, 13, 14, 15]), targets, obs_pos, echo.coef)
    weights = list(datum.loss_fn_inputs["weights"].data)
    assert all(math.isclose(w, e, abs_tol=1e-6) for w, e in zip([0, 0, 0.05, 0.05, 0.05, 0], weights, strict=True))

    # an action-masked loss (e.g. a future SDAR client) selects the complement
    class _ActionLoss(AuxiliaryLoss):
        mask = MASK_ACTION

    assert aux_positions(_ActionLoss(coef=0.1), mask) == [True, True, False, False, False, True]


def test_managed_builder_skips_empty():
    tinker = pytest.importorskip("tinker")
    from tinker.types.tensor_data import TensorData

    from rllm.trainer.tinker.aux_loss import aux_positions, build_aux_ce_datum

    mask = [1.0, 1.0, 1.0]  # all action tokens -> no observation tokens to train
    pos = aux_positions(EnvPredictionLoss(coef=0.05), mask)
    datum = build_aux_ce_datum(tinker.ModelInput.from_ints([1, 2, 3]), TensorData(data=[1, 2, 3], dtype="int64"), pos, 0.05)
    assert datum is None


# --- executor math (faithful reimplementation of agg_loss seq-mean-token-mean) ---


def _seq_mean_token_mean(neglogp_rows, mask_rows):
    """Mirror verl's agg_loss(loss_agg_mode='seq-mean-token-mean'): per-sequence
    token-mean over the mask, then mean over sequences with >0 masked tokens."""
    seq_losses, seq_valid = [], []
    for nlp, m in zip(neglogp_rows, mask_rows, strict=True):
        cnt = sum(m)
        seq_losses.append(sum(x * mm for x, mm in zip(nlp, m, strict=True)) / (cnt + 1e-8))
        seq_valid.append(1.0 if cnt > 0 else 0.0)
    denom = sum(seq_valid)
    return sum(loss * v for loss, v in zip(seq_losses, seq_valid, strict=True)) / denom if denom else 0.0


def test_inprocess_env_loss_equals_paper_l_env():
    """The verl in-process executor adds coef * mean_seq((1/|O|) sum_obs -log p);
    this guards that formula (the executor calls agg_loss with the obs mask)."""
    # seq 0: obs at idx 2,3,4 ; seq 1: obs at idx 1 (rest padding/action)
    neglogp = [[1.0, 2.0, 0.5, 0.5, 1.0, 3.0], [1.0, 0.8, 2.0, 0.0, 0.0, 0.0]]
    obs_mask = [[0, 0, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0]]
    env_ce = _seq_mean_token_mean(neglogp, obs_mask)
    paper = ((0.5 + 0.5 + 1.0) / 3 + 0.8 / 1) / 2
    # abs_tol accommodates the 1e-8 denominator epsilon agg_loss uses.
    assert math.isclose(env_ce, paper, abs_tol=1e-6)


def test_inprocess_empty_obs_is_safe():
    """A rollout with no observation tokens contributes 0 (no NaN)."""
    val = _seq_mean_token_mean([[1.0, 2.0, 3.0]], [[0, 0, 0]])
    assert val == 0.0 and math.isfinite(val)
