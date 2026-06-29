# Unified custom-loss layer

One place to define a policy loss, runs on all three backends (verl / tinker / fireworks).
Supersedes the policy-vs-aux split: an "aux" loss (ECHO) is just a term over the
observation mask. Motivating example: DPPO (arXiv:2602.04879), a GRPO variant that
replaces ratio-clipping with a per-token divergence mask.

## The contract

A loss term is a function over a normalized `LossContext`, registered with a top-level
decorator (same style as `@rllm.rollout` / `@rllm.evaluator`):

```python
import rllm

@rllm.register_loss("dppo_tv")
def dppo_tv(ctx: rllm.LossContext):
    ...
    return per_token_loss, agg_mask, metrics
```

(`rllm.register_loss` and `rllm.trainer.algorithms.register_loss` are the same object.)

`LossContext` (identical fields across backends): `pi` (current log-probs, grad),
`mu` (behavior/old log-probs — the IS denominator), `advantages`, `action_mask`,
`obs_mask`, `ref` (optional), `params`. A term returns `(per_token_loss, agg_mask,
metrics)`; the total objective is `Σ_i coef_i · Agg_mask_i(term_i(ctx))`.

Source: `rllm/trainer/algorithms/loss.py`. Built-ins: `dppo_tv`, `dppo_kl`,
`ppo_clip`, `env_prediction` (ECHO).

## Config

```yaml
algorithm:
  losses:                              # the front door; summed terms
    - {type: dppo_tv, coef: 1.0, delta: 0.2}
    - {type: env_prediction, coef: 0.05}   # ECHO — just another term
  loss_plugins: ["my_pkg.my_losses"]   # imported at startup -> fires @register_loss
  mu_source: inference                 # inference (tmax default) | proximal
```

Back-compat: `loss_fn: dppo_tv` (+ `env_loss_coef` / `aux_losses`) still works and is
resolved into terms. A backend-native `loss_fn` (verl `vanilla`, tinker `ppo`,
fireworks `grpo`) leaves `resolve_loss_terms()` empty -> the existing native path runs
unchanged.

### ECHO and the deprecated auxiliary-loss framework

There is now **one** registry. The old `rllm.trainer.algorithms.aux_loss` framework
(`@register_aux_loss`, `AuxiliaryLoss`, `EnvPredictionLoss`) is **deprecated** (imports
still work, emit `DeprecationWarning`). **ECHO is defined once** as the `env_prediction`
term:

```python
@register_loss("env_prediction", aux_mask=MASK_OBSERVATION)
def env_prediction(ctx):                 # cross-entropy on observation tokens
    return -ctx.pi, ctx.obs_mask, {}
```

`aux_mask` declares the static token region so the managed backends can realize a
CE-style additive term as an efficient extra `cross_entropy` pass (no `forward_backward_custom`
needed); verl evaluates the term directly via `CustomPPOLoss._apply_aux_losses`. Configure
ECHO via `losses: [{type: env_prediction, coef: 0.05}]`, or the back-compat
`aux_losses` / `env_loss_coef` / `adv_estimator: echo` (all resolve to the same term).

Migration: `@register_aux_loss` → `@rllm.register_loss(name, aux_mask=MASK_OBSERVATION)`;
`aux_losses:` → `losses:`.

## How each backend runs it

| Backend | Mechanism | Notes |
|---|---|---|
| **verl** | terms registered into verl's `POLICY_LOSS_REGISTRY` (`verl/custom_loss.py`, wired on Ray workers via `patch.register_rllm_custom_losses`) | verl-native `dppo_tv`/`dppo_kl` are **not** shadowed; the shim back-fills losses verl lacks (e.g. user terms). ECHO stays on `CustomPPOLoss._apply_aux_losses`. `mu = old_log_prob`. |
| **tinker** | `forward_backward_custom` closure (`tinker/custom_loss.py`) | one pass folds in all terms (surrogate + ECHO); no separate aux CE pass. `mu = sampling log-probs`. |
| **fireworks** | same `forward_backward_custom` (its client path is built on it) | `optim_step` forces `GradAccNormalization.NONE` (closure normalizes seq-mean-token-mean). `mu = sampling log-probs`. |

**Cost:** the managed (`forward_backward_custom`) path adds a forward pass (~1.5× FLOPs,
up to ~3× wall-time). verl runs in-process (cheap). The builtin server-side fast path
on tinker/fireworks cannot host custom surrogates — that's why custom losses use the
client path.

**Blackbox users:** `register_loss` + `LossContext` are public; list the defining
module under `loss_plugins`. On verl the term is cloudpickled to Ray workers, so the
module must also be importable there (set `RLLM_LOSS_PLUGINS` in the worker env).

## Limits / follow-ups

- `mu_source: proximal` not yet wired on the fireworks custom path (falls back to
  inference with a warning).
- Per-role custom losses (multi-agent) flatten to the global term set on the managed
  path; verl per-role routing is unchanged.
- Losses needing a separate teacher/privileged forward pass (SDAR-style) still need the
  `AuxiliaryLoss.requires` / AuxForward escape hatch — out of scope for the single-pass
  term model.
- End-to-end GPU/training validation of the managed trainer wiring (normalization vs
  optim_step, logprob extraction) is pending; the term math and adapters are unit-tested.
