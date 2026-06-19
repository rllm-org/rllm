# Design: Auxiliary Losses in rLLM

- **Status:** Prototype landed — migration steps 1–3 implemented on PR #668; steps 4–5 (SDAR's `AuxForward` + dynamic weights, escape hatch) pending.
- **Related:** PR #668 (ECHO, arXiv:2605.24517) — the first client of this framework
- **Scope:** the unified trainer (`UnifiedTrainer`) and its three backends — verl, tinker, fireworks

> **Implementation status.** The framework is prototyped on PR #668:
> `rllm/trainer/algorithms/aux_loss.py` (the `AuxiliaryLoss` spec, the
> `register_aux_loss` registry, `EnvPredictionLoss`, and `build_aux_losses` with
> the `env_loss_coef`/`adv_estimator=echo` back-compat sugar);
> `rllm/trainer/tinker/aux_loss.py` (the shared managed datum builder); the verl
> in-process executor in `CustomPPOLoss._apply_aux_losses`; and the tinker
> (`_get_aux_loss_futures`) / fireworks (`_build_aux_datums`) managed executors.
> ECHO is ported onto it as `env_prediction`. The `requires` field (AuxForward)
> raises `NotImplementedError` until step 4. See `tests/unified_trainer/test_aux_loss.py`.

## Summary

A growing class of agent-RL algorithms keep GRPO as the optimization backbone and
add a **token-level auxiliary loss** on top of it: ECHO (predict environment
observations), SDAR (gated on-policy self-distillation), demonstration/SFT
blending, on-policy distillation, entropy/exploration shaping, etc. Today each
such algorithm has to be hand-wired into all three backends' loss paths
separately.

This doc proposes a single abstraction — `AuxiliaryLoss` — so that adding one of
these algorithms is **one registered class plus config, with zero per-backend
loss surgery**. We refactor ECHO onto it as the first reference client and show
SDAR as the second.

## Motivation

ECHO (PR #668) is a ~10-line idea — "add a cross-entropy term on the observation
tokens" — but landing it touched three backends with bespoke code:

- `rllm/trainer/verl/verl_backend.py` → `CustomPPOLoss._add_env_loss`
- `rllm/trainer/tinker/tinker_policy_trainer.py` → `_get_env_loss_futures` / `_build_env_datum`
- `rllm/trainer/fireworks/fireworks_policy_trainer.py` → fold-and-pass in `forward_backward_from_trajectory_groups` + `optim_step`

Each backend reimplements the *same* concept: select a token subset, weight the
per-token log-probs, aggregate, and add the result to the policy loss. The
ECHO-specific knob (`algorithm.env_loss_coef`) is likewise bespoke. The next
algorithm of this family (SDAR, distillation, SFT-blend) would repeat the
3-backend surgery and add another bespoke flag.

The mechanism is general; only the *policy* was hard-coded. This doc factors the
mechanism out.

## Goals / non-goals

**Goals**

- One place to define a token-level auxiliary loss; backends pick it up generically.
- ECHO and SDAR both expressible as small declarations, sharing all backend plumbing.
- Honest, explicit handling of the verl ↔ managed-service (tinker/fireworks) asymmetry.
- Default-off: existing GRPO runs are byte-for-byte unchanged.

**Non-goals**

- Replacing the *primary* policy loss machinery (`adv_estimator`, the verl
  `POLICY_LOSS_REGISTRY`, tinker's `ppo`/`importance_sampling`). Aux losses are
  added *on top* of the policy loss.
- Forcing every conceivable loss into one mold. Losses outside the
  "token-weighted log-prob / cross-entropy" family are supported only on verl,
  via an escape hatch (see [Capability model](#capability-model)).

## Background: the two layers

An auxiliary loss decomposes into a data layer and a loss layer. Only the loss
layer is genuinely backend-specific.

| Layer | Responsibility | Backend-specific? |
|---|---|---|
| **Data** | derive per-token `(mask, target, weight-inputs)` from the merged rollout | **No** — both transforms already build the same interleaved `[A₀, obs₁, A₁, …]` row with a per-token action mask |
| **Loss** | combine forward-pass outputs (log-probs, entropy) with those tensors → scalar; add to the policy loss | **Yes** — see below |

The irreducible asymmetry in the loss layer:

- **verl** — rLLM owns the actor loss (`CustomPPOLoss`, installed via `set_loss_fn`).
  It receives the full micro-batch `data` and `model_output`, so an aux term can
  be folded into the *single existing forward/backward pass* (free, exact).
- **tinker / fireworks** — loss is computed by a fixed **server-side kernel**
  (`ppo`, `grpo`, …). rLLM cannot append a term to that kernel, but it *can*
  submit an **extra gradient-accumulated `cross_entropy` pass** over the same
  rollouts. No extra rollouts; one extra backward.

The enabling fact: **every member of the "weighted cross-entropy / log-prob over
a token subset" family maps onto `cross_entropy` on managed backends and onto an
in-pass masked sum on verl.** That family is large — env prediction (ECHO),
gated self-distillation (SDAR), SFT/demonstration blending, on-policy
distillation CE, masked-LM-style auxiliaries. Abstract *that family* and these
algorithms become declarations.

## The abstraction

### 1. Named token masks (data layer)

The shared transform attaches a small dict of named masks to each training row
(DataProto for verl; datum metadata for tinker/fireworks), computed once:

```python
aux_masks = {
    "action":      <1 on assistant/action tokens>,        # == response_mask today
    "observation": <complement of action mask, non-pad>,  # ECHO's O
    # "observation_env_only": <O excluding warning/chat-template tokens>,  # paper's O′
}
```

This is where token-subset definitions live — including refinements like the
paper's `O′` (env tokens excluding low-entropy warning/template tokens), defined
once with tokenizer knowledge instead of re-derived per backend. Aux losses
reference a mask **by name**.

> Today ECHO re-derives `(responses != pad) & ~response_mask` at the verl loss
> site and `1 - mask` in the tinker/fireworks datum builders. Those collapse into
> the single `"observation"` entry here.

### 2. The `AuxiliaryLoss` interface

```python
class AuxiliaryLoss:
    """A token-level loss added to the policy loss: coef * Agg(weight_t · term_t)."""

    mask: str                      # name of the token subset (from aux_masks)
    target: str                    # "next_token" | "sampled_token" | a named target tensor
    requires: list[AuxForward] = []  # extra forward pass(es) this loss needs (default: none)

    def weight(self, ctx: AuxContext) -> Tensor:
        """Per-token weight. Default: the static coefficient (ECHO)."""
        return ctx.coef
```

- **`mask` / `target`** — which tokens, and what the per-token term predicts.
  `next_token` ⇒ the term is `−log p_θ(x_{t+1})` (ECHO). `sampled_token` ⇒
  `−log p_θ(y_t)` on the rollout's own tokens (SDAR).
- **`weight(ctx)`** — per-token weight. Constant for ECHO; a dynamic, detached
  function of forward-pass signals for SDAR. `ctx` exposes student log-probs,
  student entropy (where available), the static `coef`, and the outputs of any
  declared `requires`.
- **`requires`** — declares extra forward passes the loss needs (e.g. SDAR's
  teacher branch). Empty for ECHO, which reuses the existing pass.

```python
class AuxForward:
    name: str                                  # key under ctx.aux[...]
    build_context: Callable                    # (x, y_<t) -> modified input (e.g. inject skills)
    needs: set[str] = {"logprobs"}             # "logprobs" | "entropy" | ...
```

### 3. The registry

Mirrors the existing `register_rllm_adv_estimator` pattern in
`rllm/trainer/algorithms/advantage.py`:

```python
AUX_LOSS_REGISTRY: dict[str, type[AuxiliaryLoss]] = {}

def register_aux_loss(name: str):
    def deco(cls): AUX_LOSS_REGISTRY[name] = cls; return cls
    return deco
```

### 4. Config

A list of specs instead of a bespoke flag:

```yaml
algorithm:
  adv_estimator: grpo
  aux_losses:
    - type: env_prediction      # registered name
      coef: 0.05
    # - type: sdar
    #   coef: 0.1
    #   gate: gap
```

`adv_estimator=echo` stays as **sugar** that injects
`aux_losses: [{type: env_prediction, coef: 0.05}]`, preserving the one-word
switch from PR #668. Bonus: aux losses **stack**.

## Executors

Each backend implements one executor that interprets *any* `AuxiliaryLoss`. These
are the current ECHO code, de-ECHO'd.

### `InProcessAuxExecutor` (verl)

Runs inside `CustomPPOLoss`, after `ppo_loss(...)`:

```python
def add(self, policy_loss, metrics, model_output, data, spec):
    ctx = AuxContext.from_verl(model_output, data, spec, self.config)  # logprobs, entropy, aux_masks
    for fwd in spec.requires:                       # e.g. SDAR teacher pass
        ctx.aux[fwd.name] = self.run_forward(fwd, data)               # detached
    w   = spec.weight(ctx)                          # [bs, resp_len]
    lp  = ctx.logprobs_on(spec.target)              # next_token or sampled_token
    term = agg_loss(loss_mat=-w * lp, loss_mask=ctx.mask(spec.mask),
                    loss_agg_mode=self.config.loss_agg_mode, **self.config.global_batch_info)
    return policy_loss + spec.coef * term, {**metrics, f"actor/aux_{spec.type}": term}
```

One forward/backward; aggregation reuses GRPO's `loss_agg_mode` + global-batch
normalization (so `coef` is on the same scale as the policy gradient).

### `ManagedPassAuxExecutor` (tinker, fireworks)

Builds a `cross_entropy` datum and submits a gradient-accumulated pass:

```python
def datums(self, raw_datums, spec, ctx):
    out = []
    for d in raw_datums:
        w = spec.weight(ctx.for_datum(d))          # client-side; uses forward()/AuxForward logprobs
        if w.any():
            out.append(Datum(model_input=d.model_input,
                             loss_fn_inputs={"target_tokens": ctx.target(d, spec.target),
                                             "weights": w * spec.coef * ctx.norm(d)}))
    return out   # submitted via forward_backward(..., "cross_entropy"), accumulated before optim_step
```

The two managed backends differ only in **normalization**, which lives here once:

- **tinker** — `optim_step` applies no normalization, so accumulation is a clean
  additive sum; `norm = 1`.
- **fireworks** — `GradAccNormalization` counts tokens/sequences across *all*
  accumulated passes, which would rescale the policy gradient. The executor folds
  the intended normalization into `weights`/advantages and forces
  `GradAccNormalization.NONE` (exactly the logic currently inline in PR #668).

The `AuxForward` (e.g. SDAR's teacher pass) is a `forward(...)` call on
context-modified datums — *the same pattern fireworks already uses for proximal
log-probs* (`_compute_proximal_logprobs`).

### `AuxContext`

The uniform view each executor builds and hands to `weight()` / target resolution:

| field | verl | tinker / fireworks |
|---|---|---|
| `logprobs` (student, per token) | `model_output["log_probs"]` | rollout / `forward()` logprobs |
| `entropy` (student, per token) | computed from logits in-pass | **only if** `forward` exposes it (see capability model) |
| `mask(name)` | from `aux_masks` in `data` | from `aux_masks` in datum meta |
| `aux[name]` | result of in-pass `AuxForward` | result of `forward()` `AuxForward` |
| `coef`, `norm` | from spec / agg config | from spec / normalization fold |

## Reference client 1 — ECHO

ECHO collapses to a declaration; all three backends run it via their executor.

```python
@register_aux_loss("env_prediction")
class EnvPredictionLoss(AuxiliaryLoss):
    mask   = "observation"     # tokens GRPO never trains
    target = "next_token"      # predict the actual environment output
    # weight() inherits the default (constant coef); requires = [] (reuses the existing pass)
```

**Before → after:**

| Before (PR #668) | After |
|---|---|
| `CustomPPOLoss._add_env_loss` (verl) | deleted → `InProcessAuxExecutor` |
| `_get_env_loss_futures` / `_build_env_datum` / `_record_env_loss_metrics` (tinker) | deleted → `ManagedPassAuxExecutor` |
| fold + env pass + `optim_step` NONE (fireworks) | deleted → `ManagedPassAuxExecutor` |
| `algorithm.env_loss_coef` | `aux_losses: [{type: env_prediction, coef}]` (echo sugar keeps the flag) |

Math is unchanged: the env term is `coef · mean_seq((1/|O|) Σ_{t∈O} −log p_θ(x_t))`
(verified numerically in PR #668 across all three backends).

## Reference client 2 — SDAR

SDAR (arXiv:2605.15155, *Self-Distilled Agentic RL*) is GRPO + a **gated
on-policy self-distillation** term. It looks unlike ECHO, but its *trainable*
gradient is again a weighted cross-entropy on a token subset:

$$\ell_t = g_t\big(\log\pi^+_\theta(y_t\mid s^+_t) - \log\pi_\theta(y_t\mid s_t)\big),\quad \text{grad}\propto -\,g_t\,\nabla_\theta\log\pi_\theta(y_t\mid s_t)$$

(the gate `g_t` and the teacher term are detached). So SDAR needs exactly two
things ECHO didn't: a **dynamic weight** (`g_t`) and an **auxiliary teacher
forward pass** over a privileged context. Both are first-class in the interface:

```python
@register_aux_loss("sdar")
class SDARLoss(AuxiliaryLoss):
    mask   = "action"          # the student's own response tokens
    target = "sampled_token"   # y_t, already in the rollout

    requires = [AuxForward(name="teacher",
                           build_context=inject_retrieved_skills,   # s⁺_t = (x, c⁺, y_<t)
                           needs={"logprobs"})]

    def weight(self, ctx):
        gap = (ctx.aux["teacher"].logprobs - ctx.logprobs).detach()      # Δ_t
        b = self.cfg.beta
        if self.cfg.gate == "gap":     return sigmoid(b * gap)
        if self.cfg.gate == "entropy": return sigmoid(b * ctx.entropy.detach())   # needs entropy
        if self.cfg.gate == "soft_or":
            e = ctx.entropy.detach()
            return sigmoid(b * (1 - (1 - e) * (1 - gap)))
```

**Per backend:**

- **verl** — the `teacher` `AuxForward` is a second in-process forward over the
  skills-augmented `input_ids`; `weight()` reads student logprobs + entropy from
  the main pass; the executor adds the gated CE. (One extra forward — SDAR is not
  "free" like ECHO; the framework makes that explicit via `requires`.)
- **tinker / fireworks** — the teacher pass is a `forward(...)` on teacher-context
  datums → teacher logprobs; the gate is computed client-side; the
  `ManagedPassAuxExecutor` submits a `cross_entropy` pass with `weights = g_t`.

The only new data-layer hook is `inject_retrieved_skills` — the teacher-context
builder — parallel to how the `observation` mask is built once.

## Capability model

The framework targets the weighted-CE/log-prob family. It must say clearly what
each backend can express, and **raise rather than silently degrade**:

| Capability | verl | tinker | fireworks |
|---|---|---|---|
| weighted CE on a named mask (ECHO) | ✅ in-pass | ✅ extra pass | ✅ extra pass |
| dynamic weight from **sampled-token** logprobs (SDAR gap gating) | ✅ | ✅ (`forward`) | ✅ (`forward`) |
| dynamic weight from **full-vocab entropy** (SDAR entropy / soft-OR) | ✅ in-pass | ⚠️ only if `forward` exposes entropy | ⚠️ only if `forward` exposes entropy |
| arbitrary Python loss on logits (value head, contrastive) | ✅ escape hatch | ❌ kernel change | ❌ kernel change |

Each `AuxForward` declares `needs` (e.g. `{"entropy"}`); a backend that can't
satisfy it raises a clear error at config time
(*"entropy gating unsupported on tinker/fireworks"*) instead of training on a
wrong signal. This makes today's implicit limitation explicit.

### Escape hatch (verl)

For losses outside the CE family, verl keeps a raw callable — `CustomPPOLoss`
already receives full `model_output` + `data`:

```yaml
aux_losses:
  - type: custom
    fn: my_pkg.losses:value_head_loss   # (model_output, data, ctx) -> (scalar, metrics)
    coef: 0.1
```

Managed backends reject `type: custom` (no in-process loss control).

## Migration / incremental plan

1. **Data layer** ✅ — named token subsets (`action`, `observation`) are resolved
   by each executor from the merged row's existing masks (no new tensor plumbing
   in this prototype; the `aux_masks`-in-transform option, including the paper's
   `O′`, remains a future refinement).
2. **Registry + executors** ✅ — `AuxiliaryLoss`, `register_aux_loss`, and the two
   executors landed; `algorithm.aux_losses` is wired into each backend.
3. **Port ECHO** ✅ — reimplemented as `EnvPredictionLoss`; `adv_estimator=echo` /
   `env_loss_coef` kept as sugar; bespoke ECHO code removed. PR #668's numerical
   checks are preserved as executor-math regression tests.
4. **Add `AuxForward` + dynamic `weight`** ⏳ — the SDAR-driven extensions; land
   gap-gated SDAR as the second client + the capability checks. (`requires`
   currently raises `NotImplementedError`.)
5. **Escape hatch + entropy capability** ⏳ — `type: custom` on verl; entropy
   output plumbing or explicit rejection on managed backends.

Steps 1–3 are a pure refactor (ECHO behavior preserved); 4–5 are additive.

## Testing strategy

- **Executor unit tests** (no GPU): given synthetic `(logprobs, mask, weight)`,
  assert each executor reproduces `coef · Agg(weight · term)` — reuse the PR #668
  numerical checks (verl seq-mean-token-mean; tinker additive; fireworks
  NONE-fold).
- **Registry/config tests**: `aux_losses` parsing, `echo` sugar expansion,
  capability errors (entropy gating on managed backends raises).
- **Equivalence test**: `aux_losses: [{type: env_prediction, coef: 0.05}]`
  produces the same loss as PR #668's `adv_estimator=echo` (guards the refactor).
- **Default-off**: empty `aux_losses` ⇒ identical to plain GRPO.

## Open questions

- **Cross-backend `coef` calibration.** verl normalizes via `loss_agg_mode` +
  global-batch info; managed backends fold normalization into weights. We aim for
  the same nominal scale, but exact parity across server kernels isn't
  guaranteed — document that `coef` may need light per-backend retuning, and
  surface `aux_*` loss metrics for monitoring.
- **Multiple `AuxForward`s sharing a pass.** If two aux losses both need the same
  teacher context, dedupe the forward. Out of scope for v1.
- **Entropy on managed backends.** Worth a small upstream ask (expose entropy
  from `forward`) to unlock entropy/soft-OR gating beyond verl.
- **Per-role aux losses.** `estimator_map`/`loss_fn_map` are per-role; aux losses
  are global in v1. Generalize later if needed.

## Appendix: why these reduce to the same executor

- **ECHO** — `target = next_token`, constant weight ⇒
  `coef · Agg_O(−log p_θ(x_t))` = length-normalized env cross-entropy.
- **SDAR** — `target = sampled_token`, detached gate weight, teacher term
  detached ⇒ trainable gradient `−Agg_A(g_t ∇ log p_θ(y_t))` = gated
  cross-entropy on the student's own action tokens.

Both are `coef · Agg_mask(weight_t · [−log p_θ(target_t)])`. The executor
implements that one expression; algorithms only supply `mask`, `target`,
`weight`, and any `requires`.
