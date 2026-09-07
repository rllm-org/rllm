# Qwen3.5-4B Native SWE GRPO (verl)

GRPO training for a **native rLLM** SWE agent — `AgentFlowEngine`, not the Harbor runtime.

|                       | `examples/harbor_swe`             | This recipe                                                        |
| --------------------- | --------------------------------- | ------------------------------------------------------------------ |
| Engine                | `RemoteAgentFlowEngine`           | `AgentFlowEngine`                                                    |
| Sandbox / agent / verifier | Harbor `Trial.run()`         | `SandboxTaskHooks` + `MiniSweAgentHarness` + `ShellScriptEvaluator`  |
| Backend               | tinker                            | **verl**                                                             |
| Train data            | `harbor:swesmith`                 | `rllm_swesmith_small` (SWE-smith slice)                              |
| Val data              | `harbor:swebench-verified`        | `swebench_verified_local` (SWE-bench Verified, local-image subset)   |

`rllm.remote_runtime.enabled=false` throughout. `harbor:swebench-verified` is used **only as a
source of task directories** (`task.toml` + `environment/Dockerfile` + `tests/test.sh`); the
Harbor runtime, Harbor agent scaffolds and `harbor:mini-swe-agent` are not involved.

Rollout path per task:

```
SandboxTaskHooks       docker run <task image> (+ agent-image mount)
MiniSweAgentHarness    mini-swe-agent CLI *inside* the sandbox
  └─ litellm ────────► rLLM model gateway (host.docker.internal) ──► vLLM (verl rollout)
GatewayManager         captures prompt/completion token ids + logprobs
ShellScriptEvaluator   /tests/test.sh → /logs/verifier/reward.txt → reward
```

## Setup

```bash
# RLLM_SCRATCH should point at a large volume: the model snapshot is ~9 GB and
# the SWE task images run to tens of GB more. Defaults to $HOME/rllm-work.
RLLM_SCRATCH=/mnt/big/rllm-work source recipe/qwen3_5_swe_grpo/env.sh

# Fused kernels for Qwen3.5's linear-attention layers -- not optional, see below.
uv pip install flash-linear-attention==0.5.2

# Backport verl PR #6660 (in v0.9.0; rLLM pins verl==0.8.0). Fixes Qwen3.5 linear
# attention under sequence packing and Ulysses SP. Idempotent; --revert undoes it.
bash recipe/qwen3_5_swe_grpo/scripts/apply_verl_patches.sh

rllm dataset pull harbor:swebench-verified            # 500 task dirs (text only)
python recipe/qwen3_5_swe_grpo/scripts/prepare_datasets.py --train-limit 24
xargs -a "$RLLM_HOME/datasets/rllm_swesmith_small/images.txt" -P 4 -I{} docker pull {}

# A larger training split for real runs. --train-name keeps it beside the smoke set
# rather than overwriting it; the image sets overlap heavily (50 tasks -> 26 images).
python recipe/qwen3_5_swe_grpo/scripts/prepare_datasets.py \
    --train-only --train-limit 50 --train-name rllm_swesmith_50
xargs -a "$RLLM_HOME/datasets/rllm_swesmith_50/images.txt" -P 4 -I{} docker pull {}
```

`prepare_datasets.py` registers two small benchmarks under `$RLLM_HOME/datasets`:

* **`swebench_verified_local/test`** — every SWE-bench Verified task whose
  `swebench/sweb.eval.x86_64.*` base image is already in the local Docker daemon. The full
  500-task set needs ~2 TB of images, so validation is pinned to what is pre-pulled. Task dirs
  are copied out of the Harbor cache so this recipe owns their timeouts.

  Pin the split with `--val-ids` (a file of task ids, or a comma-separated list). Without a pin
  the split is *whatever is currently pulled*, so landing one more base image silently changes
  what the val numbers mean; with one, a missing image is a hard error instead of a quietly
  smaller set. `recipe/qwen3_5_swe_grpo/val_tasks.txt` is the pinned ten.

  **Pick for headroom, not for scorability.** The first ten were the smallest-test-scope
  instance per repo out of the `<15 min fix` bucket -- chosen to avoid unscorable tasks, but
  that filter selects the easiest of the easy, and the *untrained* 4B policy scored 8/10 on
  them. A val set at 0.8 before training measures nothing: RL has two points of headroom and
  six steps to show them. `val_tasks.txt` rebalances across `difficulty` (3 easy / 5 medium /
  2 hard) and still oracle-screens 10/10.

  `--val-name` builds a variant side by side. Rebuilding a name deletes and recopies its task
  dirs, which pulls them out from under a run already pointing at them -- the val twin of the
  same warning on `--train-name`.
* **`rllm_swesmith_fn50/train`** — a round-robin slice of `kylemontgomery/swesmith-filtered`,
  one task per repo, so a smoke run sees repo diversity without pulling all ~4.7K images.

  **Filter by mutation kind or the run learns nothing.** SWE-smith names tasks
  `<repo>.<sha>.<kind>__<hash>`, and the kind is the only difficulty signal there is -- the
  adapter stamps `difficulty = "hard"` on every task, so `task.toml` cannot separate them. An
  unfiltered round-robin comes out ~47% `combine_file` (several bugs merged into one file). On
  such a 50-task slice the untrained 4B policy scored **0/88**: every GRPO group had reward std
  0, so `advantage/mean`, `actor/pg_loss` and `grad_norm` were all exactly `0.0` and the weights
  never moved, while the oracle scored 50/50 on the same tasks. The default `--train-kinds`
  keeps `func_pm_*` (single-line edits to one function) and `func_basic` (one function
  rewritten). `--train-pool` sets how many task dirs to download before filtering -- the filter
  discards most of the pool, so it wants to be several times `--train-limit`.

For both, the script patches `[agent].timeout_sec=3600` / `[verifier].timeout_sec=1800` into
each `task.toml` (upstream ships 3000/3000). The agent budget is SWE-Master's value
(`agent.trajectory_timeout=3600` in their RL scripts) and is sized for the 150-turn limit: the
last run measured a median of 3.9 s per turn and a p90 rollout of 567 s at 100 turns, so the
previous 900 s would have ended the clock before the turn budget on the slower tail and reported
`agent_timeout` where `max_turns_exceeded` was the truth. Both values are read at rollout time
from `task.toml`, so an already-registered dataset is retimed in place, without a rebuild:

```bash
D=$RLLM_HOME/datasets
for n in rllm_swesmith_fn50 swebench_verified_balanced; do
  find "$D/$n" -name task.toml -exec \
    sed -i '/^\[agent\]/,/^\[/ s/^timeout_sec = .*/timeout_sec = 3600.0/' {} +
done
```

Do not do that under a running job — the next rollout of every task picks the new value up.

### Screen the data with the oracle first

The oracle harness runs `solution/solve.sh` and then the real verifier, with no LLM. It is the
cheapest way to prove that a task's image, verifier and reward file all work before spending
rollouts on it:

```bash
rllm eval swebench_verified_local --agent oracle --sandbox-backend docker \
    --split test --concurrency 3 --no-ui \
    --base-url http://127.0.0.1:1/v1 --model oracle-dummy
```

A task that scores 0 here can never be solved by a policy either. `prepare_datasets.py` drops
the ones found that way — see `UNSCORABLE` (currently `astropy__astropy-7606`, whose
`PASS_TO_PASS` list names `test_compose_roundtrip[]`, a pytest param id that never appears in
the classic-style output the SWE-bench log parser reads).

Note the `--base-url`/`--model` flags: the oracle never calls an LLM, but `rllm eval` still
refuses to start without a configured provider, so point it at a dead port.

Measured on this setup: SWE-bench Verified 12/12 candidates scored 1.0 (the 10 kept plus two
`sympy` instances held in reserve), SWE-smith 1/1.

## Run

```bash
# 1 batch, 2 tasks x 4 rollouts -- "does it run?"
bash recipe/qwen3_5_swe_grpo/smoke_test.sh

# the defaults: 50 SWE-smith tasks / 8 per step = 6 steps, 1 epoch,
# validating before training, at step 5, and at the end
bash recipe/qwen3_5_swe_grpo/train_verl.sh
```

**Every run gets its own episode directory.** `train_verl.sh` stamps `RLLM_RUN_ID`
(`train.py` sets one too, so a bare `python train.py` is covered) and
`episode_log_dir` interpolates it. Without it the path stopped at
`<project>/<experiment>`, which is stable across runs: a second run of the same
experiment merged its episodes into the first run's `train_step_N_epoch_0`, and
for steps where both runs drew the same tasks in the same order the episode
filenames collided and the earlier run's files were **overwritten**. The
transcript under `logs/` carries the same id, so a log and its episodes pair up.

**Where the knobs live.** All of them are in `config/`; `train_verl.sh` only sets up the
environment, opens a transcript and execs. The split across the three YAML files is by
*owner* (recipe / rLLM / verl). One decision spans owners and has to be kept consistent by
hand: `max_prompt_length + max_response_length` (`rllm_grpo.yaml`) must equal
`max_model_len` (`verl_trainer.yaml`). Those two set the padded width of every training row
and `rllm/trainer/verl/transform.py:38` truncates an overrun *silently*, so a stale value
here does not fail — it quietly drops the tail of every long episode.

**Batch shape.** `rllm.data.train_batch_size=8` (tasks per step; verl's `sync_config`
mirrors it to `data.train_batch_size`) times `rollout.n=8` gives 64 trajectories per step,
and `ppo_mini_batch_size=8` x 8 = 64 makes that exactly one optimizer update — strictly
on-policy GRPO. That is 8 rows per GPU against `ppo_micro_batch_size_per_gpu=1`, i.e.
gradient accumulation of 8 (relevant only when scaling to an untied model, see below).

Any Hydra override passes through:

```bash
bash recipe/qwen3_5_swe_grpo/train_verl.sh \
    rllm.rollout.n=16 \
    rllm.trainer.logger="['console','wandb']" \
    recipe.val_limit=2
```

Env knobs: `RLLM_SCRATCH` (storage root), `SANDBOX_BACKEND` (default `docker`),
`RLLM_AGENT_IMAGE` (`auto` | `skip` | `repo:tag`), `MODEL_PATH`, `RLLM_RUN_DIR`,
`HF_HUB_OFFLINE`.

## What follows SWE-Master, and what does not

The RL half of this recipe is matched to SWE-Master (arXiv 2602.03411, §3.4) item by item. The
tables say where each item lives; the sections further down say why. Status is as of the tree,
not as of a measured run — the termination histogram of a real run is the check
(`batch/termination_reason/*`).

### §3.4.1 policy optimization

| paper | here | where |
| --- | --- | --- |
| leave-one-out advantage | yes | `rllm_grpo.yaml` `adv_estimator: rloo` (`n/(n-1)·(r − mean)`) |
| no std normalization | yes | `norm_adv_by_std_in_grpo: false` (inert under rloo, pinned so a revert to `grpo` does not bring it back) |
| fixed-constant length normalization (eq. 3) | yes | `loss_agg_mode: seq-mean-token-sum`, same as their scripts |
| clip-higher | yes | `eps_clip_high: 0.28` |
| no KL | yes | `kl_beta: 0.0`; the reference worker is never built |

### §3.4.2 reward design

| paper | here | where |
| --- | --- | --- |
| binary F2P+P2P reward | yes | `ShellScriptEvaluator` reads `reward.txt` |
| forced submission on budget exhaustion | yes, for free | the verifier grades the repo however the agent stopped; the docker in-container `timeout` leaves the sandbox alive |
| reward × γ for TIMEOUT / MAX_STEPS / MAX_TOKENS (eq. 4) | yes, γ = 0.5 | `config.yaml` `budget_reward_scale`; `train.py` `make_budget_scaled_grouping_hook` over `agent_timeout`, `max_turns_exceeded`, `max_prompt_length_exceeded` |
| container failure masked out (eq. 5) | yes | sandbox failure → `error` after retries → `mask_error: true` |
| verifier failure | yes, broader than the paper | `verifier_timeout` and a missing reward file are both removed |
| infrastructure beats budget | yes | `agentflow_engine._finish_episode`: an evaluator failure overwrites the agent-side reason |

The one deliberate difference: the paper's code keeps a masked rollout in the group at reward 0
and zeroes its loss; `compact_filtering` removes it from the group, so the RLOO baseline is
computed over the survivors only. See [Compact filtering](#compact-filtering-removal-not-masking).

### §3.4.3 training tricks

| paper | here | where |
| --- | --- | --- |
| environment response masking | yes, always on | `cumulative_token_mode: true`; observation tokens get mask 0 in `verl/transform.py` |
| budget awareness (steps remaining in each observation) | **no** | needs a mini-swe-agent observation-template variable; not verified for the pinned version |
| `git log` / `git show` blocked | **no** | a PATH-scoped `git` wrapper was built and then removed by decision; nothing in the tree |

### Hyperparameters, against their public RL scripts

| | SWE-Master `examples/swe/*.sh` | here |
| --- | --- | --- |
| group size `n` | 4 | 4 |
| temperature | 1.0 | 1.0 |
| lr | 1e-6 | 1e-6 |
| max prompt / response | 8192 / 122880 | 8192 / 122880 |
| max turns | 150 | 150 |
| agent timeout | 3600 s | 3600 s (`task.toml`) |
| train batch | 32 | 32 |
| `mask_truncated_samples` | False | False |
| truncated-rollout handling | `overlong_filter` (mask infra, halve budget) | `compact_filtering` + `budget_reward_scale` |

## Why the config looks like this

### Qwen3.5-4B is a hybrid-attention VLM

`Qwen3_5ForConditionalGeneration`: a vision tower plus a text stack that interleaves
`Qwen3_5GatedDeltaNet` **linear-attention** layers with full-attention layers (3:1). Two
consequences:

* **A micro-batch must hold exactly one sequence.** The recurrent layers make sequence packing
  unsafe, which pins `ppo_micro_batch_size_per_gpu=1` and `use_dynamic_bsz=false`. Raising either
  does not crash — it silently decorrelates training from the rollout. This is the one knob most
  worth understanding before tuning throughput, so it has its own section:
  [Batch shape](#batch-shape-why-micro_batch_size_per_gpu-must-stay-1).
* **`flash-linear-attention` must be installed.** 24 of the 32 layers are linear attention, and
  HF's `Qwen3_5GatedDeltaNet` falls back to a pure-PyTorch `torch_chunk_gated_delta_rule` when
  the fused kernels are missing. It is correct but slow, and it dominates the training step.
  Installing it (`uv pip install flash-linear-attention==0.5.2`, triton-only, no CUDA
  toolchain needed) is measured below.

  The companion [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d) ships as an sdist
  and needs `nvcc`, which this image does not have — so it stays uninstalled. That is fine:
  the four kernels bind **independently**
  (`self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule`), so
  the expensive delta-rule op is fused while only the cheap depthwise conv falls back to
  `F.silu(self.conv1d(...))`. **The `"The fast path is not available"` warning still prints**,
  because `is_fast_path_available` is an all-or-nothing `all(...)` over all four — ignore it
  and check `perf/throughput` instead.

verl 0.8.0 does support the architecture (`verl/models/transformers/qwen3_5.py`, plus a
`monkey_patch.py` branch for `model_type in ("qwen3_5", "qwen3_5_moe")`).

#### The vision tower and the MTP head are carried, not trained

The checkpoint is 4.66B parameters in three groups, and only one of them is what this recipe
trains:

| group | params | share | in this run |
| --- | --- | --- | --- |
| `model.language_model` | 4,205,751,296 | 90.3% | trained |
| `model.visual` | 333,514,240 | 7.2% | carried, never updated |
| `mtp` | 120,599,552 | 2.6% | never instantiated |

**The vision tower is not frozen, and verl's switch for it does nothing.**
`actor.freeze_vision_tower` exists in `verl/trainer/config/actor/actor.yaml:52` (default
`false`) and as a dataclass field at `verl/workers/config/actor.py:160`, but **no code in the
installed packages reads it** — setting it `true` is a no-op in verl 0.8.0. Instantiating
`Qwen3_5ForConditionalGeneration` confirms every `model.visual.*` parameter comes up with
`requires_grad=True`.

It does not matter much here, because SWE tasks carry no images. With no image tokens the
tower is never called, so it stays out of the autograd graph, `p.grad` stays `None`, and
torch's AdamW skips those parameters *and* never allocates their state (the state dict is
populated lazily in `step()`). So the 2.5 GiB of Adam moments never appear and the weights
do not move. What remains is the bf16 weights themselves: 0.62 GiB, or **0.08 GiB per GPU**
sharded across 8 — noise against the 30-40 GB this run actually peaks at. If you do want it
gone, the flag will not do it; it takes a `model.visual.requires_grad_(False)` after model
construction.

The `mtp` group is different: `Qwen3_5ForConditionalGeneration` does not build it at all, so
those 120.6M parameters exist in the checkpoint on disk and nowhere in the training graph.
They load as unexpected keys and are dropped. Harmless for training, but worth knowing when
saving and reloading — a checkpoint written from this run carries the MTP head at its
original values while the language model has moved.

### preserve_thinking, and why it is not just a chat template

Qwen3.5 uses **interleaved thinking**: its chat template keeps `<think>` blocks only for turns
after the last user query and strips the rest. `rllm.gateway.cumulative_token_mode=true` avoids
re-tokenizing history by extending the previous turn's raw token ids, and the `renderers` bridge
refuses to do that whenever the policy would have re-rendered — which, under the default
`tool_cycle` retention, is **every new user message**. mini-swe-agent feeds each shell result
back as a user message, so without `preserve_thinking` every single turn falls back to a full
re-render: historical reasoning dropped, prompt growing O(T²), and the prefix-extension chain
broken into one batch row per turn.

```
qwen3.5  {}                            bridge=None (full re-render)
qwen3.6  {}                            bridge=None (full re-render)
qwen3.6  {preserve_thinking: True}     bridge=OK, prefix-extension holds
```

Hence `renderer_family: qwen3.6` + `renderer_kwargs: {preserve_thinking: true}`.
`Qwen36Renderer` subclasses `Qwen35Renderer` and differs only in tool-argument JSON
serialization and this flag, so it is token-identical for this workload. (The Qwen3.6-27B
*Jinja* template likewise differs from Qwen3.5-4B's in exactly two lines — the
`preserve_thinking` gate and the `tojson` change — so shipping it as
`custom_chat_template` buys nothing here and is left out.)

Health check for all of this: `training/rollout_actor_probs_pearson_corr` in the step metrics.
It compares the actor's recomputed log-probs against the ones sampled during rollout, and sits
at **0.9995** on this recipe. A low value means the token stream that reached training is not
the one the policy sampled.

### Tool calling

mini-swe-agent v2 requires the model to answer with a **`bash` tool call** ("Your response MUST
include AT LEAST ONE bash tool call"), and its litellm client always sends `tool_choice="auto"`.
vLLM rejects that unless auto tool choice is on, so the rollout gets
`enable_auto_tool_choice=true` + `tool_call_parser=qwen3_xml` (which matches Qwen3.5's
`<tool_call><function=..><parameter=..>` template markup) alongside `reasoning_parser=qwen3`.

### Batch sizes

verl asserts `data.train_batch_size * rollout.n % n_gpus == 0` (FSDP `minimal_bsz = n_gpus`),
and rLLM multiplies `actor.ppo_mini_batch_size` by `rollout.n` to get the mini-batch of
trajectories. With 8 GPUs:

| | tasks/batch | `rollout.n` | trajectories | `ppo_mini_batch_size` | updates/batch |
|---|---|---|---|---|---|
| `train_verl.sh` | 4 | 8 | 32 | 4 | 1 (on-policy) |
| `smoke_test.sh` | 2 | 4 | 8  | 2 | 1 |

### Verifying the renderer / cumulative-token path

The whole multi-turn story rests on three things being simultaneously true, and
none of them fails loudly. `scripts/verify_cumulative.py` checks all three against
*real* rollout tokens rather than a synthetic fixture:

```bash
bash recipe/qwen3_5_swe_grpo/smoke_test.sh \
    rllm.gateway.store=sqlite \
    rllm.gateway.db_path="$RLLM_SCRATCH/traces/verify.db"
python recipe/qwen3_5_swe_grpo/scripts/verify_cumulative.py
```

| check | what it asserts | measured |
| --- | --- | --- |
| `enable_thinking` | the generation prompt opens `<think>` and the completion closes it | 10/10, 8/8, 15/15, 9/9 turns |
| prefix extension | turn N's `prompt_ids` literally begins with turn N-1's `prompt_ids + completion_ids` | 9/9, 7/7, 14/14, 8/8 transitions |
| `preserve_thinking` | earlier turns' `<think>` blocks are still in turn N's prompt | 15-turn session: 15 `<think>` / 14 `</think>` in the final prompt |
| loss mask | replaying `transform.py`'s merge, `mask==1` covers exactly the sampled completions | token **ids** match, not just counts |

`enable_thinking` needs no configuration: Qwen3.5-4B's own template already ends the
generation prompt with `<think>\n`, so the renderer auto-resolves it to `True`. What
does need configuring is `preserve_thinking` — see above.

The loss-mask check is worth reading the output of, because counts alone can agree by
accident. It compares the actual token ids, and prints what each side decodes to:

```
mask=1 head: 'The user wants me to fix an issue where the `exceptions` property of ...'
mask=0 head: '\n<|im_start|>user\n<tool_response>\n{\n  "returncode": 0, ...'
```

Model-generated tokens are trained on; shell output is not.

### Batch shape: why `micro_batch_size_per_gpu` must stay 1

Short version: **1 is not a memory concession you can trade away, it is a correctness
requirement**, and raising it is not even faster.

#### With `use_remove_padding=true` (our config) — runs, silently wrong

verl's packed forward flattens the whole micro-batch into a single row
(`verl/workers/engine/fsdp/transformer_impl.py`):

```python
input_ids_rmpad = input_ids.values().unsqueeze(0)   # (1, total_nnz) -- every sequence concatenated
```

Sequence boundaries then survive only in `position_ids`. Full-attention layers recover them via
`cu_seqlens`; the 24 gated-delta-net layers cannot:

```python
# transformers/models/qwen3_5/modeling_qwen3_5.py
def forward(self, hidden_states, cache_params=None, attention_mask=None):   # no cu_seqlens
    ...
    self.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, ...)     # none passed either
```

fla's `chunk_gated_delta_rule` *does* accept `cu_seqlens` — HF simply never supplies it. So a
recurrent layer carries state straight from the end of one sample into the start of the next.

Measured A/B, one step, identical config apart from the micro-batch:

| | micro=1 | micro=2 |
| --- | --- | --- |
| `training/rollout_actor_probs_pearson_corr` | 0.999761 | **0.969670** |
| `training/rollout_probs_diff_mean` | 0.002748 | **0.016976** |
| `training/rollout_probs_diff_max` | 0.229 | **0.999968** |
| `timing_s/update_actor` | 69.4 s | 55.6 s |
| `perf/throughput` | 193 tok/s | 203 tok/s |

No exception, no warning; the run completes and reports a loss. What breaks is the thing GRPO
depends on: the actor's recomputed log-probs no longer match what the policy actually sampled. A
`probs_diff_max` of ~1.0 means some token's probability is off by the entire range. The payment
for that is ~5% throughput.

The same packing setting is read by all three forwards — old-log-prob, reference, and the actor
update all consume the output of `left_right_2_no_padding`, and `use_remove_padding` lives on the
*model* config shared by every worker (`transformer_impl.py:125`). So
`ref.log_prob_micro_batch_size_per_gpu` and `rollout.log_prob_micro_batch_size_per_gpu` are bound
by the same rule, not just the actor's knob.

#### With `use_remove_padding=false` — safe in principle, unusable in practice

The other branch pads instead of packing:

```python
input_ids = torch.nested.to_padded_tensor(input_ids, ...)        # (B, max_seq_len)
attention_mask = build_attention_mask_from_nested(...)           # real per-row mask
```

Each row is independent, and `apply_mask_to_padding_states()` at the top of the gated-delta-net
forward zeroes the padded positions — so `micro_batch > 1` here would be numerically fine. It
still does not work for this model, for two measured reasons:

1. **It is incompatible with the fused kernels.** `use_remove_padding=false` +
   `use_fused_kernels=true` dies at `verl/workers/utils/padding.py:131` —
   `assert sequence_offsets[-1].item() == values.shape[0]`, i.e. rLLM's `no_padding_2_padding`
   received model output whose leading dimension is not the packed token count. (Observed, not
   traced further — the combination was abandoned rather than debugged.)
2. **Without the fused kernels it OOMs on logits.** Unfused, the head materializes
   `B x T x 248320 x 4 bytes`. At `B=1` and this recipe's sequence lengths that is already a
   single 21.48 GiB allocation that fails on an 80 GB card. `B=2` doubles it, and padding sets
   `T` to the batch maximum for *every* row rather than each row's own length — so the padded
   path is strictly worse than packing at this vocabulary size, before any correctness
   argument.

#### Summary

| config | correct? | works? | note |
| --- | --- | --- | --- |
| `remove_padding=true`, micro=1 | ✅ | ✅ | **what this recipe uses** |
| `remove_padding=true`, micro>1 | ❌ silently | ✅ runs | pearson 0.9998 → 0.9697, for ~5% throughput |
| `remove_padding=false`, micro=1 | ✅ | ❌ | OOM at this recipe's lengths (21.48 GiB logits alloc) |
| `remove_padding=false`, micro>1 | ✅ | ❌ | assert with fused kernels; worse OOM without |

To actually raise the micro-batch you would have to teach HF's `Qwen3_5GatedDeltaNet` to thread
`cu_seqlens` (derivable from `position_ids`) into fla's kernel, which already accepts it. Until
then, throughput comes from `flash-linear-attention` and the fused head, not from batch shape.

### Memory: the vocabulary is the wall

Qwen3.5's vocabulary is **248320 tokens**. Materializing logits for a single merged trajectory
of ~23K tokens in fp32 is one **21 GB** allocation, and with FSDP param/optimizer offload
already in play an 80 GB card still dies with
`torch.OutOfMemory: Tried to allocate 21.48 GiB`. Neither micro-batch size nor offloading helps
— it is one tensor for one sequence.

`actor_rollout_ref.model.use_fused_kernels=true` +
`fused_kernel_options.impl_backend=torch` is the fix: verl's `FusedLinearForPPO` computes
`log_probs` / `entropy` in chunks straight off the hidden states and never builds the logits
tensor (verl has a Qwen3.5-specific `forward_with_torch_backend`). Everything else — sequence
length, `ppo_micro_batch_size_per_gpu`, offload — is a second-order knob next to this one.

The fused path consumes the **packed** layout, so `use_remove_padding=true` comes with it;
without it rLLM's `no_padding_2_padding` fails on
`assert sequence_offsets[-1].item() == values.shape[0]`. That is safe here only because a
micro-batch holds exactly one sequence — see the hybrid-attention note above.

### Long context, sequence parallelism, and FSDP version

These three interact, so they were measured together. Summary: **fsdp2 + a large
`max_model_len` gets you to 128K on its own. Sequence parallelism is broken on verl
0.8.0 and repaired by a backported patch, but even repaired it is not worth enabling
at this model size — carry it for when the model outgrows one card.**

#### fsdp2 is strictly better here — it is the default

`fsdp` (FSDP1, `FSDP(module, ...)`, FlatParameter) vs `fsdp2` (`fully_shard`,
per-parameter DTensor sharding), same config otherwise, one step:

| | fsdp | fsdp2 |
| --- | --- | --- |
| `rollout_actor_probs_pearson_corr` | 0.999662 | **0.999745** |
| `perf/max_memory_allocated_gb` | 27.1 | **22.8** |
| `perf/throughput` | 210 tok/s | **215 tok/s** |
| `timing_s/update_actor` | 62.8 s | **46.9 s** |
| `timing_s/old_log_probs` | 24.9 s | **13.9 s** |
| FSDP1 `state_dict_type` deprecation warnings | many | **none** |

verl also patches this model's vision tower specifically for the fsdp2 cpu-offload
path (`monkey_patch.py`: *"Step 2: patch vision model to fix fsdp2 cpu_offload bug"*),
so fsdp2 is the path upstream expects for Qwen3.5.

#### Context length: 128K fits without sequence parallelism

Activation memory for one sequence, fwd+bwd, measured with
`scripts/mem_scaling.py` (single GPU, params unsharded — the real run shards and
offloads them, so subtract ~8 GB):

| seq_len | peak GB | activation GB | GB / 1K tok |
| --- | --- | --- | --- |
| 32768 | 23.7 | 8.0 | 0.251 |
| 65536 | 34.9 | 19.3 | 0.301 |
| 131072 | 60.7 | 45.1 | 0.352 |

End-to-end at `max_model_len=131072`, `max_response_length=122880`,
`agent_step_limit=100`:

```
response_length/max            73542      one merged row of 73.5K tokens
batch/n_turns                  64.7       merge_compression_ratio identical -> still one row
perf/max_memory_allocated_gb   36.2       reserved 40.1, on 80 GB cards
pearson_corr                   0.999707   correctness holds at length
maximum-context 400s           0
termination_reason/error       0.000
```

So 64K is exercised for real and 128K has headroom. Throughput drops (169 vs 215 tok/s)
because the sequences are longer, which is the expected trade rather than a problem.

#### Both of those are fixed by a backported verl patch

verl **v0.9.0** carries [PR #6660](https://github.com/verl-project/verl/pull/6660),
*"[model, fsdp] fix: support Qwen3.5 linear attention under Ulysses SP"* (merged
2026-07-20), whose opening sentence describes exactly the two problems measured below:

> Current Qwen3.5/Qwen3.5-MoE training with FSDP remove-padding + Ulysses SP does not pass
> packed sequence boundaries into linear attention, so packed samples can share
> linear-attention state across sequence boundaries; the Qwen3.5 fused PPO loss can also
> re-slice already SP-sliced labels.

rLLM pins `verl==0.8.0` (upstream `rllm-org/rllm` does too), and v0.9.0 makes the V1 PPO
trainer the default — which collides with the verl internals rLLM imports — so the fix is
**backported** rather than picked up by upgrading:

```bash
bash recipe/qwen3_5_swe_grpo/scripts/apply_verl_patches.sh          # idempotent
bash recipe/qwen3_5_swe_grpo/scripts/apply_verl_patches.sh --check  # report state
bash recipe/qwen3_5_swe_grpo/scripts/apply_verl_patches.sh --revert # undo
```

`patches/verl-pr6660-qwen3_5-ulysses-sp.patch` touches three verl files plus the PR's own
distributed regression test. It stays clear of the PPO trainer, which is why it grafts onto
0.8.0 cleanly.

One adjustment was needed for our stack: the PR was validated against transformers
5.3.0.dev0, where `cache_params.has_previous_state` is a property. In 5.5.3 it is
`has_previous_state(layer_idx)`. Inert during training (`cache_params` is always `None`
under `use_cache=False`) but wrong for a cached forward, so the backport calls it correctly.
The PR is otherwise defensive about version drift — it introspects with
`inspect.signature` before passing new kwargs, which is why the rest applied unchanged.

Verification on this stack (transformers 5.5.3, FLA 0.5.2 — both newer than the PR's):

```
torchrun --nproc_per_node=2 -m pytest tests/special_distributed/ \
    test_qwen35_linear_attention_ulysses_sp.py          ->  2 passed
```

The test compares a full forward against the Ulysses-SP forward on boundary-aligned,
sequence-cut, single-sequence and many-short packed cases, asserting
`grad_err_ratio < 2e-3`. Both the `causal_conv1d_fn` and torch-fallback paths pass.

Note the packing half of the fix is **not** gated on SP —
`if packed_cu_seqlens is not None and pass_packed_cu_seqlens:` in
`transformer_impl.py`, with `use_ulysses_sp` only adjusting the SP padding. So it also
lifts the `ppo_micro_batch_size_per_gpu=1` constraint documented above. The engagement flag
is derived by introspection (`"cu_seqlens" in signature(module.forward).parameters`); all
three `forward_with_*_backend` variants declare it, so it does turn on.

#### What was measured before the patch

Two independent reasons, both measured.

**It crashes.** `ulysses_sp=2` dies in the fused head:

```
flash_attn/ops/triton/cross_entropy.py:171
    assert labels.shape == (n_rows,)
```

`patch_vlm_for_ulysses_input_slicing` slices `inputs_embeds` with
`padding=False`, while verl's `forward_with_torch_backend` pads *then* slices the
labels via `ulysses_pad_and_slice_inputs`. The two disagree whenever `total_nnz` is
not divisible by the SP size.

**And it would be wrong anyway.** Ulysses gives each rank a contiguous slice of the
sequence. Full attention recovers the global view through its all-to-all; the 24
`Qwen3_5GatedDeltaNet` layers cannot, because HF calls
`chunk_gated_delta_rule(..., initial_state=None)` — every rank restarts the recurrence
from zero. Measured on the op directly, two ranks over a 1024-token sequence:

| second chunk computed with | max abs error | relative |
| --- | --- | --- |
| `initial_state=None` (what ulysses gives it) | 0.006767 | **10.9%** |
| `cp_context` from `fla.ops.cp.context.get_cp_cu_seqlens` | 0.000000 | **0.00%** |

The fix is exactly what PR #6660 does: `flash-linear-attention` ships a real
context-parallel gated delta rule, and the patch builds an `FLACPContext` from verl's
ulysses process group (handling the depthwise conv boundary too, via
`conv_kernel_size`) and reconciles the label padding.

> A trap worth recording: constructing `FLACPContext(...)` by hand and passing it is a
> **silent no-op**. fla's CP path only engages in varlen mode, and the per-rank metadata
> has to come from `get_cp_cu_seqlens()`. A first attempt at this reproduced the same
> 10.9% error with `cp_context` supplied, which looks like "the fix doesn't work" rather
> than "the fix wasn't wired".

#### What it looks like after the patch

Same one-step measurement, `ulysses_sequence_parallel_size=2`:

| | sp=1 | sp=2 |
| --- | --- | --- |
| result | ok | **ok** (was: crash) |
| `rollout_actor_probs_pearson_corr` | 0.999745 | **0.999605** |
| `rollout_probs_diff_max` | 0.229 | 0.167 |
| `perf/max_memory_allocated_gb` | 22.8 | **15.9** |
| `perf/throughput` | 215 tok/s | 153 tok/s |
| `timing_s/update_actor` | 46.9 s | 146.5 s |

So SP is now **correct** — and for a 4B model at ~37K sequences, still not worth turning
on: it buys 30% of activation memory for 3.1x the actor update time, and memory is not
the binding constraint here. `train_verl.sh` therefore keeps
`ulysses_sequence_parallel_size=1`. The reason to carry the patch is that the trade
inverts once activation memory *is* the constraint — a larger model, or context past
what one card holds.

`ppo_micro_batch_size_per_gpu=2` is likewise correct now (pearson 0.999412) but slower,
so that default stands too — for a different reason than before. Normalized against token
count (both runs processed ~710K tokens and did identical work: 32 rollouts, 32 mini-batch
rows, one update):

| | µs / token | vs micro=1 |
| --- | --- | --- |
| `ppo_micro_batch_size_per_gpu=1` | 64.0 | — |
| `ppo_micro_batch_size_per_gpu=2` | 128.6 | **2.01x** |
| `ulysses_sequence_parallel_size=2` | 198.2 | 3.10x |

The kernels are **not** the cause. `scripts/pack_cost.py` times the real model's fwd+bwd
under the four layouts that matter and finds no packing penalty at all:

```
B / A  = 1.00x   turning on the varlen path (cu_seqlens=[0,2T]) costs nothing
C / D  = 0.97x   two rows packed is marginally *faster* than two separate passes
A/(2E) = 1.03x   attention is not quadratic across a packed row
```

So the 2x lives somewhere in verl's micro-batch plumbing, not in the model. It was not
isolated further — the knob stays at 1 either way, which is both correct and fastest.

The patch is safe to leave applied. It routes the default path through the packed
kernels too (a single sequence becomes `cu_seqlens=[0, T]`), so the default config was
re-measured for regression — there is none:

| | before patch | after patch |
| --- | --- | --- |
| `rollout_actor_probs_pearson_corr` | 0.999745 | 0.999384 |
| `rollout_probs_diff_max` | 0.229 | 0.174 |
| `perf/max_memory_allocated_gb` | 22.8 | 22.4 |
| `perf/throughput` | 215 tok/s | 220 tok/s |
| `timing_s/update_actor` | 46.9 s | 45.2 s |

All within run-to-run variation across different rollout content.

#### Before scaling to a larger model

Two things to check, neither of which bites Qwen3.5-4B.

**verl [#7520](https://github.com/verl-project/verl/issues/7520)** (open, unassigned):
an **untied** `lm_head` + FSDP2 + `use_fused_kernels=true` + gradient accumulation crashes
with `AttributeError: 'FSDPParam' object has no attribute '_unsharded_param'` during the
backward of a non-final micro-batch. Qwen3.5-4B sets `tie_word_embeddings: true` so it is
unaffected, but 9B and up are untied and would hit exactly this recipe's configuration.
Until it is fixed, a larger model needs either no gradient accumulation
(`ppo_mini_batch_size * rollout.n == the whole batch`, which is what this recipe already
does) or fused kernels off — and fused kernels are not really optional at this vocabulary
size, so prefer the former.

**Sequence parallelism becomes worth its cost.** At 4B it buys 30% of activation memory
for 3.1x the actor update. Once activation memory is what binds — a bigger model, or
context past one card — that inverts, and the backported patch is what makes it available.

### Compact filtering: removal, not masking

The config keys are named `mask_*`, but nothing is masked — the episode is dropped
(`rllm/trainer/algorithms/transform.py:120`):

```python
for episode in episodes:
    termination_reason = episode.termination_reason or TerminationReason.UNKNOWN
    if compact_filtering_config and compact_filtering_config.should_mask(termination_reason):
        continue                       # episode never becomes a trajectory
```

This runs *before* `TrajectoryGroup`s are built, which has a consequence worth being explicit
about: **a filtered rollout does not contribute to its GRPO group's mean and std either.** GRPO
normalizes reward within a group, so removing members changes the baseline the survivors are
scored against. At `rollout.n=8` with two rollouts filtered, the remaining six form the group.
That is usually what you want — a timed-out rollout is not evidence about the policy — but it is
not the same thing as "its loss is zeroed".

Token-level loss masking is a **separate**, always-on mechanism
(`rllm/trainer/verl/transform.py:385`):

```python
seg["mask"].extend([0] * len(delta_obs))   # shell output — not the policy's tokens
seg["mask"].extend([1] * len(action))      # what the model generated
```

When the turns of an episode merge into one row, observation tokens get mask 0 and action tokens
get mask 1, regardless of how the episode ended.

#### Which reasons are dropped, and which are kept at half reward (SWE-Master)

SWE-Master (arXiv 2602.03411, §3.4.2 and §6.3) sorts terminations into two families and the
recipe follows it. The split is what `rllm_grpo.yaml`'s `compact_filtering` block and
`config.yaml`'s `budget_reward_scale` encode between them:

| family | reasons | what happens |
| --- | --- | --- |
| budget exhaustion | `max_turns_exceeded`, `max_prompt_length_exceeded`, `agent_timeout` | **kept**. The verifier grades whatever the agent left in the repo (the paper's "forced submission" — free here, because `ShellScriptEvaluator` runs `tests/test.sh` no matter how the agent stopped, and the docker backend's in-container `timeout` leaves the sandbox alive), then `train.py` multiplies the reward by `budget_reward_scale` (0.5, the paper's value). |
| infrastructure | `verifier_timeout`, `error` | **dropped** by `compact_filtering`. The reward is not a measurement of anything. |

The paper's reason for not dropping the first family (§6.3): the DeepSWE recipe of zeroing and
masking truncated rollouts collapsed their training, and it also silently removes the hardest
tasks from the curriculum, since those are the ones that run out of budget.

Two things had to be true for the table to hold on this native path:

* `agent_timeout` needs a producer. `cli_harness.run` used to swallow the sandbox exec timeout
  as a warning and return `None`, so an agent that ran out of its `[agent] timeout_sec` was
  indistinguishable from one that finished — `ENV_DONE`. It now returns an empty Episode
  stamped `AGENT_TIMEOUT`, which enrichment preserves.
* `timeout` had to split. The single `TIMEOUT` reason was produced by the verifier on this path
  (infrastructure) and by the agent/trial clock on the Harbor and Workflow paths (budget), so one
  `mask_timeout` flag could not do the right thing for both. It is now `VERIFIER_TIMEOUT` /
  `AGENT_TIMEOUT`, with `mask_verifier_timeout` / `mask_agent_timeout`.

The engine also orders the stamps as *infrastructure beats budget beats default*: an evaluator
failure overwrites whatever the agent side reported, because an agent that timed out and whose
grader then also timed out has no usable reward, and the filter has to see the failure that
makes it unusable.

#### There are three removal stages, and the first one does most of the work

| stage | where | condition |
| --- | --- | --- |
| 1 | `rllm/engine/agent_workflow_engine.py:262` | `all(len(t.steps) == 0 ...)` → "has no valid trajectories, dropping it from the batch" |
| 2 | `rllm/trainer/algorithms/transform.py:120` | `compact_filtering.should_mask(termination_reason)` → episode dropped |
| 3 | `rllm/trainer/buffer.py:203` | `len(g.trajectories) < min_trajs_per_group` → whole **group** dropped |

**In every run measured here, stage 2 removed nothing.** `num_trajs_before_filter` equalled
`num_trajs_after_filter`, and the termination histogram was all `env_done` plus some `error` —
`max_response_length_exceeded`, `timeout` and `max_turns_exceeded` were **0.000 throughout**. So
the `mask_timeout` / `mask_max_response_length_exceeded` paths were configured but unexercised.
(That was before the reasons were actually produced on the native path and before `timeout` was
split into `agent_timeout` / `verifier_timeout` — see the next section.)

What actually happened to context overruns, traced through `train3` (`max_model_len=32768`,
64 rollouts over 2 steps):

```
44  vLLM HTTP 400 "maximum context length is 32768 tokens"
43  EnrichMismatchError (traces=N agent_steps=0 empty_prompt_ids=1 ...)
25  Attempt 1/2 failed   →  7 recovered on retry
18  Attempt 2/2 failed   →  18 episodes with zero steps
18  "has no valid trajectories, dropping it from the batch"     ← stage 1
    termination_reason/error: 0.125 (step 1) + 0.4375 (step 2) = 18/64 ✓
```

An overrun is **not** a graceful truncation. vLLM rejects the request, litellm's retries all fail
against the same ceiling, the agent produces no usable step, and the episode is discarded whole
before compact filtering is ever consulted. That is why `max_model_len` is sized to the training
row width rather than left to `compact_filtering` to clean up — see
[Turn budget](#turn-budget) and [Lengths](#lengths).

### Picking `agent_step_limit` and `max_model_len` together

These two are one decision. Every mini-swe-agent turn appends its observation and action
to a prompt that is never re-rendered, so the cumulative prompt grows roughly linearly and
the ceiling has to be sized for the *last* turn, not the first.

Measured across runs (`response_length/max` is one merged training row):

| `agent_step_limit` | `max_model_len` | turns reached | longest merged row | 400s |
| --- | --- | --- | --- | --- |
| 25 | 32768 | 25.0 | 18,557 | 0 |
| 40 | **32768** | ~40 | — | **44** ← lost 18 episodes |
| 40 | 49152 | 36–40 | 41,686 | 0 |
| 100 | 131072 | 64.7 | 73,542 | 0 |

About **1,100 tokens per turn** of merged response, so:

```
max_response_length  ≈  1150 × agent_step_limit
max_model_len         =  max_prompt_length + max_response_length
```

Two rules behind that arithmetic:

* **`max_model_len` must be ≥ the training row width.** Going over it is not a graceful
  truncation — vLLM returns HTTP 400 mid-rollout, litellm's retries all hit the same wall,
  and the episode is discarded whole (see the compact-filtering section). Going *under* the
  row width just means compact filtering masks the overflow, which is the survivable
  failure. So err generous.
* **Being generous is cheap.** Activation memory runs 0.25–0.35 GB per 1K tokens, so even
  131072 lands at ~47 GB per 80 GB card. Throughput is what you actually pay: 220 tok/s at
  49152 versus 169 at 131072, because the sequences are longer.

**Defaults in this recipe:** `agent_step_limit=40`, `max_model_len=49152`
(= 8192 + 40960). Validated over six runs with zero context-length errors. That is the
tightest pairing that holds — 32768 at the same step limit lost 28% of its rollouts.

Raising the turn budget means raising both, e.g. `agent_step_limit=60` wants
`max_response_length≈69632`, `max_model_len=77824`.

### Turn budget

Upstream mini-swe-agent defaults to `agent.step_limit: 0` (unlimited) and relies on
`cost_limit` instead — which is inert here, since the gateway-routed model has no litellm cost
table and the harness sets `MSWEA_COST_TRACKING=ignore_errors`. Left unbounded, the cumulative
prompt keeps growing until vLLM rejects a turn with *"This model's maximum context length is N
tokens"*; the agent's retries all fail and rLLM discards the episode with an
`EnrichMismatchError` instead of scoring it.

`train.py` therefore wraps the harness in `StepLimitedMiniSweAgent`, which passes
`-c mini.yaml -c agent.step_limit=<recipe.agent_step_limit>` (naming `mini.yaml` again is
required: `-c` *replaces* the default config rather than layering on it). Size the limit against
`max_model_len` — roughly **620 prompt tokens per turn** measured on SWE-smith with this policy,
so 50 turns ≈ 33K tokens under the 40960 ceiling.

### Lengths

An AgentFlow batch row is one **merged trajectory**: the prompt is turn 1's prompt and the
response accumulates every later turn's observation + action delta. So `max_response_length`
(24576) has to cover a whole episode, while `rllm.rollout.train.max_tokens` (4096) caps a single
turn. Rollouts that end badly rather than finishing are removed by
[compact filtering](#compact-filtering-removal-not-masking).

### HuggingFace rate limits

Eight FSDP workers, eight vLLM servers and the gateway each resolve `Qwen/Qwen3.5-4B` on
startup. Unauthenticated that exceeds the Hub's 500-requests / 300s per-IP budget, and the
throttled process reports it as `OSError: Unable to load vocabulary from file` — a rate-limit
error wearing a corrupted-cache costume. `train_verl.sh` therefore defaults to
`HF_HUB_OFFLINE=1` (the snapshot is already in `$HF_HOME` by launch time). Export `HF_TOKEN`,
or `HF_HUB_OFFLINE=0`, if you actually want the hub consulted.

## What a healthy step looks like

`train_verl.sh` defaults on 8x A100-80GB — 4 SWE-smith tasks x `rollout.n=8`, untrained
Qwen3.5-4B. Learning-signal numbers are from a step that happened to contain a solve;
performance numbers are from a steady-state step (second step of a run, triton kernels cached).

**Is the plumbing right?**

| metric | why it matters | observed |
| --- | --- | --- |
| `training/rollout_actor_probs_pearson_corr` | actor's recomputed log-probs vs. what the policy actually sampled. The single best check that cumulative-token mode, `preserve_thinking` and the trace pipeline all line up. | **0.9997** (`probs_diff_mean` 0.0026) |
| `batch/merge_compression_ratio` | turns folded into one batch row by prefix extension. Equal to `batch/n_turns` means the whole episode merged; **1.0 means prefix extension is broken** and every turn became its own row. | **36.4** (= `n_turns`) |
| `batch/termination_reason/error` | rollouts that died instead of finishing — usually the context ceiling. | **0.0** (was 0.125 at `max_model_len=32768`) |

**Is there a learning signal?**

| metric | why it matters | observed |
| --- | --- | --- |
| `reward/mini-swe-agent/{mean,max,std}` | is the verifier reachable at all by the policy? | 0.031 / 1.0 / 0.17 |
| `batch/mini-swe-agent/fractions/{too_hard,effective}` | groups with no reward spread vs. groups that carry signal. | 0.75 / **0.25** |
| `advantage/mini-swe-agent/std` | non-zero means GRPO has something to push on. | **0.50** (max 2.65, min -0.38) |
| `actor/pg_loss`, `grad_norm` | the actual update. | **0.031**, **0.312** |

`pg_loss = 0` with `grad_norm = 0` and `too_hard = 1.0` is **not** a broken optimizer: GRPO
normalizes reward within a group, so a group where every rollout scores 0 contributes exactly
zero advantage. That is a normal first-step picture; the gradient appears as soon as one rollout
in a group solves its task. If it never does, the training set is too hard for the starting
policy rather than misconfigured.

**Is it fast enough?** A clean A/B — same config, only `flash-linear-attention` differs:

| | torch fallback | with fla | |
| --- | --- | --- | --- |
| `timing_s/update_actor` | 344.3 s over 718K tokens | 62.8 s over 947K tokens | |
| ⤷ per token | 479 µs | **66 µs** | **7.2x** |
| `timing_s/old_log_probs` | 34.3 s | 24.9 s | |
| `perf/throughput` (incl. rollout) | 120 tok/s | **210 tok/s** | 1.75x |
| `perf/max_memory_allocated_gb` | 41.9 | **27.1** | |
| `rollout_actor_probs_pearson_corr` | 0.999602 | **0.999662** | |

Two things worth noting. First, the **first** step after installing fla is slower than steady
state (`update_actor` 145 s) while triton compiles and autotunes the kernels; the cache is on
disk, so it is a one-time cost per machine. Second, the fused path does not just run faster —
it agrees with vLLM's rollout slightly *better* than the torch fallback does. A standalone
forward comparison of the two implementations shows ~0.28 max logit difference on a bf16 model
(different reduction order), which sounds alarming in isolation; the pearson metric is the one
that answers whether it matters, and it moved the right way.

## Agent image mount

`RLLM_AGENT_IMAGE=auto` (the default) bakes mini-swe-agent into a small local image once and
mounts it read-only at `/opt/rllm/agent` in every task sandbox (Docker `type=image` mount), so
no rollout pays for `uv tool install`. Same mechanism as `rllm eval --agent-image`.

* Docker backend only. `modal` / `daytona` fall back to a per-task install.
* `RLLM_AGENT_IMAGE=skip` forces the per-task install path.
* Building the image needs network on the *host*; the sandbox itself does not install anything.

## Sandbox network isolation — still not supported

`DockerSandbox` / `ModalSandbox` / `DaytonaSandbox` expose no `network_mode` or egress policy,
and the task sandboxes here **do** need the network:

| setting | internet | gateway (LLM) | result |
|---|---|---|---|
| `network_mode=none` | blocked | **blocked** | every rollout fails — the agent's LLM calls go out through `host.docker.internal` |
| default bridge (today) | allowed | allowed | reward hacking is possible (web search, external APIs, `pip install`) |
| selective egress (not implemented) | blocked | host-only allow | what you actually want |

Both verifiers also need the network at runtime: SWE-bench Verified's `test.sh` runs
`uv run parser.py` (pulls `swebench`, `datasets`, `fastcore`), and SWE-smith's runs
`apt-get install` + `uv add pytest swebench datasets swesmith`. Blocking egress means
pre-baking those into the task images first.

### A hung verifier used to hang the whole run

`rllm/sandbox/backends/docker.py` accepted `exec(timeout=...)` and dropped it —
the docstring said so ("currently unused by Docker SDK"). So `task.toml`'s
`[verifier] timeout_sec` was a no-op on the docker backend and `exec_run`
blocked forever. One agent patch sent a verifier `pytest` into a loop and the
training run sat for **2h58m** at 63/64 of a rollout step, GPUs at 0%, that
container at **519 GB RSS**. Nothing in the stack could end it.

`exec()` now enforces the timeout in two layers:

1. `timeout --kill-after=15s <sec>` inside the container. Cheap, and the
   container survives, so the episode is simply scored unsolved.
2. A client-side deadline. This is the layer that actually guarantees the
   caller unblocks. In the incident a `timeout` process was already `<defunct>`
   while its *grandchild* pytest kept running — GNU `timeout` signals only its
   direct child. Killing the container is the only thing that reliably reaps
   every descendant, so on expiry that is what happens.

Note the exit codes: GNU `timeout` returns 124 when SIGTERM sufficed but **137**
when it escalated to `--kill-after`'s SIGKILL, and 137 is also what an OOM kill
looks like. Elapsed time separates them, so a real OOM still surfaces as a
command failure rather than a timeout.

Containers were also created with no limits at all, which is why 519 GB was
reachable. `_sandbox_resource_kwargs` now maps them for docker too, mirroring
what Harbor's own compose applies — `deploy.resources.limits` sets `cpus` and
`memory` and nothing else, so this sets `nano_cpus` and `mem_limit` and nothing
else. Values come from the task's `[environment]`, and an undeclared field falls
back to Harbor's `EnvironmentConfig` default (1 cpu, 2048 MB) rather than to
"unlimited", so rLLM and Harbor run the same task under the same budget.

Two wrinkles in reading those values:

* **Both spellings exist.** Harbor canonicalizes on `memory_mb` / `storage_mb`
  and keeps `memory` / `storage` as deprecated string fields. swesmith and
  swebench-pro task dirs write `memory_mb = 4096`; swebench-verified writes
  `memory = '4G'`. `_parse_size_mb` reads the string form with the same units
  Harbor accepts (G/M/K) and, like Harbor, raises on anything else instead of
  guessing — a size written wrong should be visible. This fixed the remote
  backends too: daytona previously read only `memory_mb`/`storage_mb`, so every
  swebench-verified task silently got no memory or disk limit.
* **`storage_mb` is not applied to docker.** It would need `--storage-opt size=`,
  i.e. overlay2 on xfs with pquota, unavailable under the containerd snapshotter.

Note that `cpus` is a CFS quota, not a core mask: a single-process pytest is
unaffected, a parallel test run is slowed to the declared budget. That is the
budget Harbor already runs these tasks under.

## Fixes this recipe required in rLLM

Each of these was a hard failure of the native training path, not a tuning choice:

1. **`rllm/trainer/unified_trainer.py`** — the training warm queue created sandboxes without the
   agent-image mount while `SandboxTaskHooks` still marked the CLI as pre-provisioned, so every
   rollout died with `mini-swe-agent: command not found` (zero traces, reward 0). Now passes
   `agent_mount_image=`, mirroring `rllm/eval/runner.py`.
2. **`rllm-model-gateway` `proxy.py`** — cumulative token mode rewrites turn 2+ to
   `/v1/completions`, which has no reasoning or tool-call parser, and the old code handed the raw
   text back as `content`. A tool-calling agent then saw no `tool_calls` on any turn past the
   first and stalled after ~4 turns. The cumulative handler now parses the completion with the
   same renderer that built the prompt (`_completion_to_chat_message`), recovering `content`,
   `reasoning_content` and `tool_calls`. Turn count went 4 → 34 on the same task.

   The same applies to **streaming**: these tool calls are markup, so they cannot be recognised
   from a partial text delta without a per-family incremental parser. A streaming request that
   carries `tools` is therefore routed to `_handle_cumulative_buffered_stream`, which takes the
   completion non-streaming, runs it through the shared `_cumulative_completion` core, and
   re-emits it as well-formed `chat.completion.chunk` frames. The client keeps its SSE contract
   and gets the same message the JSON path would return; only the incrementality is lost.
   Requests without tools keep the true incremental path.
   Covered by `tests/integration/test_cumulative_tool_calls.py` (JSON + SSE).
3. **`rllm-model-gateway` `models.py` / `server.py` + `rllm/gateway/manager.py` +
   `rllm/trainer/config/rllm/base.yaml`** — new `gateway.renderer_kwargs`, so a renderer family's
   config can be overridden (`preserve_thinking`). Previously only the bare family name could be
   set, and `config_from_name("qwen3.6")` defaults `preserve_thinking=False`.
4. **`train.py`** uses `DatasetRegistry.load_dataset(..., as_tasks=True)`. Without it every row
   becomes a `Task` rooted at `dataset_dir="."` and verifier auto-detection fails with
   `No verifier configured for task ...`.

## Files

```
recipe/qwen3_5_swe_grpo/
├── README.md
├── env.sh                        # RLLM_SCRATCH / HF_HOME / RLLM_HOME / venv
├── train.py                      # Hydra entry → unified AgentTrainer(agent_flow=..., backend="verl")
├── train_verl.sh                 # launcher: env, transcript, exec (no knobs)
├── smoke_test.sh                 # 1 batch, minimal everything
├── config/
│   ├── config.yaml               # Hydra entry + recipe.* (datasets, turn budget, reward scale, sandbox)
│   ├── rllm_grpo.yaml            # the `rllm.*` tree (data, sampling, gateway, GRPO)
│   └── verl_trainer.yaml         # verl-native (model, FSDP2 actor/ref, vLLM rollout)
├── patches/
│   └── verl-pr6660-...patch      # backported verl fix (see Setup)
└── scripts/
    ├── prepare_datasets.py       # builds + registers the train/val benchmarks
    ├── apply_verl_patches.sh     # applies patches/ to the installed verl
    ├── verify_cumulative.py      # asserts the renderer / cumulative-token invariants
    ├── mem_scaling.py            # activation memory vs sequence length
    └── pack_cost.py              # fwd+bwd cost of packed vs separate micro-batches
```
