"""Tests for the (deprecated) auxiliary-loss shim and its delegation to the unified
loss-term system.

ECHO is now the ``env_prediction`` term (rllm.trainer.algorithms.loss); the old
``AuxiliaryLoss``/``EnvPredictionLoss`` API is deprecated. ``build_aux_losses`` /
``resolve_additive_terms`` resolve ``aux_losses`` / ``env_loss_coef`` into unified
``ResolvedTerm`` additive terms. The managed datum builder and the verl executor math
are unchanged.

See design/unified-custom-loss.md.
"""

import math
import warnings

import pytest
from omegaconf import OmegaConf

from rllm.trainer.algorithms import AlgorithmConfig
from rllm.trainer.algorithms.aux_loss import AuxiliaryLoss, build_aux_losses
from rllm.trainer.algorithms.loss import (
    MASK_ACTION,
    MASK_OBSERVATION,
    RLLM_LOSS_REGISTRY,
    ResolvedTerm,
    get_loss,
    get_term_aux_mask,
    resolve_additive_terms,
)


def _algo(**kw):
    base = {"adv_estimator": "grpo", "norm_adv_by_std_in_grpo": True}
    base.update(kw)
    return AlgorithmConfig.from_config(OmegaConf.create(base))


# --- ECHO is now a single term ----------------------------------------------


def test_echo_is_a_single_term():
    assert "env_prediction" in RLLM_LOSS_REGISTRY
    assert get_term_aux_mask("env_prediction") == MASK_OBSERVATION


def test_old_aux_api_is_deprecated():
    with pytest.warns(DeprecationWarning):

        class _Custom(AuxiliaryLoss):
            mask = MASK_OBSERVATION

        _Custom(coef=0.1)


# --- config resolution (build_aux_losses delegates to resolve_additive_terms) ---


def test_echo_sugar_yields_env_prediction_term():
    terms = build_aux_losses(_algo(adv_estimator="echo"))
    assert [(t.name, t.coef, t.mask) for t in terms] == [("env_prediction", 0.05, MASK_OBSERVATION)]
    assert terms[0].fn is get_loss("env_prediction")


def test_grpo_builds_no_additive_terms():
    assert build_aux_losses(_algo()) == []


def test_explicit_aux_losses_list():
    terms = resolve_additive_terms(_algo(aux_losses=[{"type": "env_prediction", "coef": 0.02}]))
    assert len(terms) == 1 and terms[0].coef == 0.02 and terms[0].mask == MASK_OBSERVATION


def test_explicit_list_takes_precedence_over_sugar():
    terms = resolve_additive_terms(_algo(adv_estimator="echo", aux_losses=[{"type": "env_prediction", "coef": 0.03}]))
    assert len(terms) == 1 and terms[0].coef == 0.03


def test_env_loss_coef_on_grpo_enables_env_prediction():
    terms = resolve_additive_terms(_algo(env_loss_coef=0.05))
    assert len(terms) == 1 and terms[0].name == "env_prediction" and terms[0].coef == 0.05


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        resolve_additive_terms(_algo(aux_losses=[{"type": "nope", "coef": 0.1}]))


def test_missing_coef_raises():
    with pytest.raises(ValueError):
        resolve_additive_terms(_algo(aux_losses=[{"type": "env_prediction"}]))


# --- managed-backend datum builder (tinker / fireworks) ---------------------


def _echo_term(coef=0.05):
    return ResolvedTerm(name="env_prediction", fn=get_loss("env_prediction"), coef=coef, params={}, mask=MASK_OBSERVATION)


def test_managed_aux_positions_and_weights():
    tinker = pytest.importorskip("tinker")
    from tinker.types.tensor_data import TensorData

    from rllm.trainer.tinker.aux_loss import aux_positions, build_aux_ce_datum

    mask = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # actions: 0,1,5 ; observations: 2,3,4
    echo = _echo_term()
    obs_pos = aux_positions(echo, mask)
    assert obs_pos == [False, False, True, True, True, False]

    targets = TensorData(data=[10, 11, 12, 13, 14, 15], dtype="int64")
    datum = build_aux_ce_datum(tinker.ModelInput.from_ints([10, 11, 12, 13, 14, 15]), targets, obs_pos, echo.coef)
    weights = list(datum.loss_fn_inputs["weights"].data)
    assert all(math.isclose(w, e, abs_tol=1e-6) for w, e in zip([0, 0, 0.05, 0.05, 0.05, 0], weights, strict=True))

    # an action-masked additive term selects the complement
    action_term = ResolvedTerm(name="_act", fn=get_loss("env_prediction"), coef=0.1, params={}, mask=MASK_ACTION)
    assert aux_positions(action_term, mask) == [True, True, False, False, False, True]


def test_managed_builder_skips_empty():
    tinker = pytest.importorskip("tinker")
    from tinker.types.tensor_data import TensorData

    from rllm.trainer.tinker.aux_loss import aux_positions, build_aux_ce_datum

    mask = [1.0, 1.0, 1.0]  # all action tokens -> no observation tokens to train
    pos = aux_positions(_echo_term(), mask)
    datum = build_aux_ce_datum(tinker.ModelInput.from_ints([1, 2, 3]), TensorData(data=[1, 2, 3], dtype="int64"), pos, 0.05)
    assert datum is None


# --- executor math (faithful reimplementation of agg_loss seq-mean-token-mean) ---


def _seq_mean_token_mean(neglogp_rows, mask_rows):
    seq_losses, seq_valid = [], []
    for nlp, m in zip(neglogp_rows, mask_rows, strict=True):
        cnt = sum(m)
        seq_losses.append(sum(x * mm for x, mm in zip(nlp, m, strict=True)) / (cnt + 1e-8))
        seq_valid.append(1.0 if cnt > 0 else 0.0)
    denom = sum(seq_valid)
    return sum(loss * v for loss, v in zip(seq_losses, seq_valid, strict=True)) / denom if denom else 0.0


def test_inprocess_env_loss_equals_paper_l_env():
    neglogp = [[1.0, 2.0, 0.5, 0.5, 1.0, 3.0], [1.0, 0.8, 2.0, 0.0, 0.0, 0.0]]
    obs_mask = [[0, 0, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0]]
    env_ce = _seq_mean_token_mean(neglogp, obs_mask)
    paper = ((0.5 + 0.5 + 1.0) / 3 + 0.8 / 1) / 2
    assert math.isclose(env_ce, paper, abs_tol=1e-6)


def test_inprocess_empty_obs_is_safe():
    val = _seq_mean_token_mean([[1.0, 2.0, 3.0]], [[0, 0, 0]])
    assert val == 0.0 and math.isfinite(val)
