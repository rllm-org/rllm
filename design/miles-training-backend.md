# Miles as an rLLM training backend

Plan for adding `backend="miles"` alongside `verl` / `tinker` / `fireworks`, with
rLLM's `UnifiedTrainer` owning the loop and Miles owning the GPUs.

Miles: https://github.com/radixark/miles (slime fork; SGLang rollout + Megatron-LM/FSDP2 training).
References below are to a local checkout at `~/miles`.

## 0. Status (2026-08-21)

Branch `feat/miles-backend`. Landed and tested off-GPU:

| Piece | State |
|---|---|
| Phase 0 dependency analysis | done — see below |
| Miles importable without its container image | **done** — recipe in §0.1 |
| Config bridge validated against Miles' real `parse_args` | **done** — 7 live tests |
| Transform validated against real `convert_samples_to_train_data` | **done** — 5 live tests |
| `pyproject.toml`: tinker optional, `miles` extra | done |
| Table-driven launcher dispatch + install hints | done |
| `rllm/trainer/algorithms/step_merge.py` (shared merge helpers) | done, tinker rewired to it |
| `rllm/trainer/miles/miles_config.py` (config bridge) | done, 34 tests |
| `rllm/trainer/miles/transform.py` (episodes → Samples) | done, 19 tests |
| `rllm/trainer/miles/_flag_audit.py` (offline flag-drift guard) | done |
| `rllm/trainer/config/rllm/backend/miles.yaml` | done |
| `miles_engine.py` (SGLang /generate, TITO) | done |
| `miles_backend.py` (BackendProtocol) | done, 12 tests |
| `miles_launcher.py` (Ray init + bring-up) | done |
| `patch.py` (advantages CP-slice) | done, contract asserted against installed miles |
| `custom_loss.py` | **not needed** — stock `policy_loss_function` reads `batch["advantages"]`, verified live |
| End-to-end training run | **done** — 8 GRPO steps on countdown, Qwen3-1.7B, 8xH100, val pass@1 0.444 |

Zero regressions: the failing-test set is byte-identical before and after
(91 pre-existing failures in a venv without ray/verl/vllm, including two float32
round-trip failures in `tests/unified_trainer/test_tinker_transform.py` that
predate this branch).

### 0.1 Miles without the container image

The image is **not** required for Phases 0–3. Miles installs from source; what its
`requirements.txt` omits (torch, sglang, megatron-core, transformer_engine) comes
from the image base, and only `sglang` is needed for the modules this backend touches.

```bash
# a separate venv: miles pins transformers==5.12.1, which breaks tinker in the rLLM venv
uv venv --python 3.12 /path/to/mvenv
cd ~/miles
# nvidia-resiliency-ext 0.6.0 ships only manylinux_2_39 wheels (glibc 2.39 / Ubuntu 24.04).
# On glibc 2.35 there is no sdist to fall back on. Fault-tolerance only -> drop it.
grep -v '^nvidia-resiliency-ext' requirements.txt > /tmp/reqs.txt
uv pip install --python /path/to/mvenv --prerelease=allow -r /tmp/reqs.txt   # ~36s
uv pip install --python /path/to/mvenv --no-deps -e .
uv pip install --python /path/to/mvenv --prerelease=allow "sglang==0.5.16"   # ~48s
uv pip install --python /path/to/mvenv -e /path/to/rllm
```

That makes all five modules the backend touches importable — `miles.utils.types`,
`miles.utils.arguments`, `miles.ray.rollout.train_data_conversion`,
`miles.backends.training_utils.data`, `miles.ray.placement_group` — with **no
Megatron and no TransformerEngine**, because the FSDP path imports neither.

Two caveats:

- Stock `sglang==0.5.16` downgrades torch 2.13 → 2.11. Fine for argument parsing and
  the CPU-side seams; the image's forked SGLang is authoritative for anything
  behavioral.
- **This box cannot run GPU work in either venv.** The driver is CUDA 12.8
  (570.195.03) while both venvs carry cu130-built torch, so
  `torch.cuda.is_available()` is False — true of the pre-existing rLLM venv too.
  Phase 1 onward needs cu12 wheels, or Miles' `cu12-x86` image variant
  (`ENABLE_CUDA_13=0`).

### 0.2 End-to-end run (validated 2026-08-21)

`examples/countdown/unified_trainer/train_countdown_unified_miles.sh` — countdown,
Qwen3-1.7B, 4 train GPUs (FSDP) + 4 rollout GPUs, thinking disabled, 8 GRPO steps.
Completes in a few minutes; final `val/countdown/pass@1` 0.444 with 8 `actor_train`
passes. Reward is noisy across 8 steps (0.22-0.53) and does not visibly climb -- this
validates the **mechanism**, not learning.

Environment fixes this needed on a CUDA-12.8 box, in order:

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` false | `torch==2.11.0+cu128` (+ matching torchvision/torchaudio) from the pytorch cu128 index; sglang pins the torch *version*, not its CUDA build |
| `libnvrtc.so.13` / `libcudart.so.13` missing | `sglang-kernel` from `docs.sglang.ai/whl/cu129/`; uninstall `sgl-deep-gemm` (cu13-only, and DeepGEMM needs SM100+ anyway) |
| `FileNotFoundError: 'ninja'` | `pip install ninja`; put the venv bin and `/usr/local/cuda/bin` on PATH so Ray workers inherit them |
| `__triton_launcher...so: cannot open shared object file` | `TRITON_CACHE_DIR=$(mktemp -d /dev/shm/triton.XXXXXX)` -- and `TRITON_` had to be added to rLLM's forwarded env prefixes or workers never see it |
| `FlashAttention2 ... doesn't seem to be installed` | `miles.attn_implementation=sdpa` (FA2 ships in Miles' image wheels) |
| `404 /begin_weight_update` | stock PyPI sglang serves generation fine but has no weight-update endpoint; install the fork: `git clone -b sglang-miles`, `pip install --no-deps --no-build-isolation -e python` |
| `ModuleNotFoundError: megatron` inside `compute_log_probs` | `pip install megatron-core`. The FSDP path *does* need it -- Miles' shared log-prob kernel uses Megatron's fused cross-entropy (there is a `TODO` there for a fallback) |

## 1. Target architecture

One Ray cluster, one driver process (rLLM), inside Miles' docker image.

```
rllm train backend=miles
  └─ MilesTrainerLauncher
       ├─ build Miles argparse Namespace from OmegaConf          (config bridge)
       ├─ ray.init(runtime_env=...)  +  create_placement_groups(args)
       ├─ create_rollout_manager(args, pgs["rollout"])           # SGLang fleet + router
       │     rollout fn = sleep_rollout  → never called
       ├─ create_training_models(args, pgs, rollout_manager)     # actor_model: RayTrainGroup
       └─ UnifiedTrainer(backend_cls=MilesBackend)
            ├─ init_rollout_engine        → MilesEngine → SGLang router /generate (TITO)
            ├─ generate_episodes          → rLLM workflow engine (unchanged)
            ├─ transform_to_backend_batch → Episodes → list[Sample] → train_data → data_ref
            ├─ process_backend_batch      → no-op (Miles' actor does its own fwd passes)
            ├─ compute_advantages         → rLLM native, per-token, rides in the batch
            ├─ update_policy              → actor_model.train(rollout_id, {"data_ref": ...})
            └─ on_batch_end               → save_model() ; update_weights()
```

Miles' `train.py` / `train_async.py` are not used. `RolloutManager` is kept purely as the
engine-fleet + router + weight-update broker — `update_weights` routes through it
(`miles/ray/actor_group.py:119` → `rollout_manager.get_updatable_engines_and_lock`), which is
why we keep it rather than launching SGLang ourselves.

## 2. Deliverables

```
rllm/trainer/miles/
  __init__.py
  miles_launcher.py     # MilesTrainerLauncher: config bridge + Ray/PG/actor bring-up
  miles_config.py       # OmegaConf -> miles argparse Namespace
  miles_backend.py      # MilesBackend(BackendProtocol[Iterable, MilesBatch])
  transform.py          # Episodes/TrajectoryGroups -> list[Sample] -> train_data dict
  custom_loss.py        # rLLM loss -> miles loss-fn signature shim (trainer ranks)
  patch.py              # worker-side monkey patches applied to miles
  utils.py              # sync_config shared-keys table
rllm/engine/rollout/miles_engine.py
rllm/trainer/config/rllm/backend/miles.yaml
```

Touched: `rllm/trainer/unified_trainer.py` — one `elif backend == "miles"` branch (~10 lines,
mirroring the tinker branch at :1163).

## 3. The four hard problems, and the decisions

### P1 — Config bridge (the biggest chunk)

Miles args are one flat `argparse.Namespace` built on top of **Megatron's own** `parse_args`
(`miles/utils/arguments.py:2664`), so the whole Megatron flag surface shares one CLI and
`PYTHONPATH=/root/Megatron-LM` must be set before import.

**Decision:** `miles_config.build_miles_args(cfg) -> Namespace` renders an argv list and calls
Miles' own `parse_args()` under a temporarily swapped `sys.argv`. Do *not* hand-construct a
Namespace — `miles_validate_args` + `megatron_validate_args` + `set_default_megatron_args` do
substantial derivation we want to run.

Two config layers, matching how tinker/verl already work:

- `rllm/trainer/config/rllm/backend/miles.yaml` holds a `miles:` block that mirrors Miles flags
  near-verbatim (`actor_num_nodes`, `rollout_num_gpus`, `train_backend`, `megatron_model_type`,
  `hf_checkpoint`, `ref_load`, …), rendered to `--kebab-case` argv.
- A `_SHARED_KEYS` table in `rllm/trainer/miles/utils.py` fed to the existing
  `sync_shared_keys` (see `rllm/trainer/tinker/utils.py`) mirrors `rllm.data.*` /
  `rllm.algorithm.*` onto Miles names so users set them once.

Escape hatch: `miles.extra_args: ["--foo", "bar"]` appended verbatim, so an unmirrored Miles
flag is never a blocker.

Pinned by us, not user-settable (assert in `validate_config`):

| flag | value | why |
|---|---|---|
| `--rollout-function-path` | `miles.rollout.sleep_rollout.sleep` | rLLM owns generation; RolloutManager must never generate |
| `--rollout-global-dataset` | `false` | rLLM owns the dataset; skips Miles' Dataset/tokenizer load |
| `--disable-compute-advantages-and-returns` | set | advantages come from rLLM |
| `--loss-type` | `custom_loss` (phase 3+) | with `--custom-loss-function-path rllm.trainer.miles.custom_loss:loss_fn` |
| `--colocate` | forbidden (phase 1) | disaggregated only, so generation and training don't fight over GPUs |

### P2 — Per-token advantages into the trainer

Miles computes advantages **inside the train step** from scalar rewards
(`miles/backends/megatron_utils/actor.py:558` → `miles/backends/training_utils/loss.py:28`),
gated on `args.compute_advantages_and_returns`. rLLM computes its own per-token advantages on
the driver.

`--disable-compute-advantages-and-returns` (`miles/utils/arguments.py:1437`) turns Miles'
computation off — its help text says "useful for sft or custom loss function". So the only
missing piece is transport for a driver-supplied, response-aligned float array.

That transport already exists for `rollout_log_probs`: `get_rollout_data`
(`miles/backends/training_utils/data.py:93`) CP-slices exactly three keys through
`slice_log_prob_with_cp`, which asserts `len(x) == response_length` — the same shape our
advantages have.

**Decision:** patch that key tuple worker-side from rLLM to include `"advantages"`. Same
mechanism rLLM already uses for verl (`rllm/trainer/verl/patch.py`, applied via
`runtime_env.worker_process_setup_hook` in `ray_runtime_env.py`). ~5 lines.

Consequence worth noting: with advantages arriving CP-correct, **stock Miles
`policy_loss_function` works unmodified** — it reads `batch["advantages"]` at
`miles/backends/training_utils/loss_hub/losses.py:91`. So phases 1–2 need no custom loss at
all; `custom_loss.py` is only for rLLM-specific losses (DPPO etc.).

Test CP=1 first (`C_i == R_i`, slicing is identity), then CP>1.

**Two further sites, found only by running end to end.** With either one unpatched the
run completes and looks healthy while the loss quietly uses Miles' own advantages:

1. `_package_shards` (`train_data_conversion.py:357`) copies a **hardcoded allowlist**
   of keys into each DP shard, so any extra key the driver attaches is dropped.
2. The Megatron actor gates `compute_advantages_and_returns` on the flag; the **FSDP
   actor calls it unconditionally** (`fsdp_utils/actor.py:496`), recomputing from the
   scalar rewards and overwriting what rLLM shipped. Patching this needs the *actor
   module's* binding repointed, not just the source module's -- both actors import the
   function by value.

Verified live: the worker receives 16 advantage rows per DP rank (64 total) and the
short-circuit keeps rLLM's GRPO z-scores (-1.732 = -sqrt(3), 0.5773 = 1/sqrt(3)).

Upstream as a small PR (`for key in (..., "advantages")` plus a `ROLLOUT_DATA_VALUE_SPEC`
entry) so the patch can be dropped.

### P3 — Rollout engine

`MilesEngine(RolloutEngine)` modelled directly on `VerlEngine`
(`rllm/engine/rollout/verl_engine.py`): `supports_token_in_token_out = True`,
`get_token_output_from_token_input` POSTs to SGLang's raw `/generate` on Miles' router
(`args.sglang_router_ip` / `sglang_router_port`) with `input_ids` and `return_logprob=True`.

Raw `/generate` rather than Miles' TITO session server, because rLLM already owns
tokenization, prompt assembly and loss masks end to end (`ChatTemplateParser` + the transform).
Routing through the session server would install a second tokenizer authority and duplicate the
merge logic. Miles documents this path explicitly (`docs/user-guide/generate-endpoint.md`:
"owning prompt construction, tokens, and loss masks").

Field mapping is already 1:1:

| rLLM `ModelOutput` | Miles `Sample` |
|---|---|
| `prompt_ids` + `completion_ids` | `tokens` |
| merged completion length | `response_length` |
| `logprobs` | `rollout_log_probs` |
| `routing_matrices` | `rollout_routed_experts` (R3) |
| `weight_version` | `weight_versions` |

### P4 — Transform

`transform.py` ports `trajectory_to_datums` (`rllm/trainer/tinker/transform.py:78`) — the
prefix-extension merge, lineage partitioning and per-token advantage broadcast are reusable
verbatim. It is *simpler* than the tinker version: Miles takes raw `tokens` and does its own
shift internally, so drop `create_rightshifted_model_input_and_leftshifted_targets`.

One `Sample` per merged sequence:

- `tokens` = `[O1, A1, O2, A2, …]`
- `response_length` = `len(tokens) - len(O1)`
- `loss_mask` = 0 on observation tokens, 1 on action tokens (length `response_length`)
- `rollout_log_probs`, `advantages` = same length, 0-filled over observation tokens
- `status` from `Episode.termination_reason` → `COMPLETED` / `TRUNCATED` / `ABORTED`
- `index` / `group_index` from `TrajectoryGroup.group_id`

Then `convert_samples_to_train_data(args, samples, metadata, None, None)` +
`split_train_data_by_dp(...)` (`miles/ray/rollout/train_data_conversion.py`) produce the
`{"sample_indices": …, "data_ref": …}` pack `RayTrainGroup.train` expects
(`miles/ray/actor_group.py:77`).

Ray's object store ignores `value_spec` (`miles/utils/object_store.py:117` is just
`ray.put(value)`), so the extra `advantages` key needs no spec entry — *unless* someone enables
the Mooncake store, where it does. Covered by the same upstream PR.

## 4. Phases

**Phase 0 — env feasibility. Dependency half done; image half still open.**

Measured, not guessed. The blocker is exact: `tinker-cookbook 0.5.2` requires
`transformers!=5.4.*,!=5.5.0,!=5.5.1,!=5.5.2,!=5.5.3,<=5.5.4,>=4.57.6`, while Miles
requires `transformers==5.12.1`. rLLM listed `tinker` / `tinker-cookbook` in **core**
`dependencies` *and* in the `tinker` extra, so every install inherited that cap.

Resolved by dropping the redundant core copy. Proven end-to-end: `uv pip install -e .`
into a `transformers==5.12.1` venv now leaves it at 5.12.1, whereas adding
`tinker-cookbook` back downgrades it to 5.5.4. Also verified by blocking `tinker` and
`tinker_cookbook` at the import hook: `rllm`, `rllm.cli.main`,
`rllm.trainer.unified_trainer`, `rllm.data`, `rllm.hooks`, `rllm.gateway.manager` and
the algorithm modules all import clean — the only gap was `ray` (the verl extra).
`rllm/trainer/fireworks/fireworks_policy_trainer.py` imports tinker at module scope,
so the `fireworks` extra now declares it explicitly.

Still open, and it needs a GPU box with docker (this dev box has neither): Miles'
image is mandatory regardless of the pin. It builds off `lmsysorg/sglang:v0.5.16`
with a forked SGLang (`sglang-miles`), a forked Megatron-LM
(`radixark/Megatron-LM@miles-main`) and prebuilt cu130 wheels — none of it
assemblable from PyPI. So `backend=miles` means "runs in the Miles image", and
`validate_config` should say so.

*Exit:* `import rllm`, `import miles`, `import megatron` in one interpreter, inside
the image.

**Phase 1 — train-only loop, no rLLM generation (2–3 days).**
Bring up placement groups + `RolloutManager(sleep)` + actor from the rLLM launcher. Feed a
hand-built `list[Sample]` fixture straight into `actor_model.train()`.
*Exit:* loss decreases on a memorized batch; `update_weights()` completes.

**Phase 2 — MilesEngine + transform (3–5 days).**
Point rLLM's workflow engine at the Miles router. Single-turn GRPO on a small dense model
(Qwen3-4B, `--train-backend fsdp` to skip Megatron checkpoint conversion).
*Exit:* reward curve on a math task matches the verl backend within noise.

**Phase 3 — advantages passthrough + custom loss (2–3 days).**
Apply the `data.py` patch and `--disable-compute-advantages-and-returns`.
*Exit:* identical loss/grad-norm to Phase 2 when rLLM is configured to reproduce Miles' GRPO;
then a non-native rLLM loss runs through `custom_loss.py`.

**Phase 4 — multi-turn + Megatron + scale (1–2 weeks).**
Merged multi-turn samples, CP>1 slicing, `--train-backend megatron` with TP/PP/EP, checkpoint
save/resume via `save_model` + `start_rollout_id`, R3 routing replay.
*Exit:* an agentic run (terminal-bench / SWE) on a ≥30B MoE.

**Phase 5 — async (deferred).**
Map `rllm.async_training` onto `on_policy_updated` + `update_weights_interval`, with
`pause_generation` / `continue_generation` (`miles/backends/sglang_utils/sglang_engine.py:623`)
around weight updates. Do not start until 1–4 are stable.

Roughly 3–4 weeks to Phase 4 for one engineer.

## 5. Things that will bite

1. **Ray ownership.** Both frameworks call `ray.init` and build placement groups. rLLM inits
   Ray first with Miles' runtime env (`MEGATRON_` / `SGLANG_` prefixes are already in rLLM's
   `FORWARD_PREFIXES`), then hands `pgs` to Miles' `create_*`. Also: Miles' `execute_train`
   normally runs a `pkill -9 sglang; ray stop --force` preamble
   (`miles/utils/external_utils/command_utils.py:166`); we bypass it, so stale SGLang processes
   become our problem.
2. **We use Miles against its grain.** It ships as a driver (`ray job submit train.py`), not a
   library. `create_placement_groups` / `create_rollout_manager` / `create_training_models` are
   plain importable functions, but carry no API-stability promise. Pin a Miles commit; expect
   breakage on bumps. Budget for upstreaming 2–3 small patches (advantages key, value spec,
   maybe an args-from-dict entry point) to shrink the surface we depend on.
3. **Megatron checkpoint conversion.** `--train-backend megatron` needs an offline
   HF→`torch_dist` conversion for `ref_load`. Phase 2 sidesteps it with FSDP; Phase 4 owns it.
4. **`sample.index` / `group_index`.** `convert_samples_to_train_data` still uses these for
   grouping and metrics even when we supply advantages. Set them from `TrajectoryGroup.group_id`
   or reward normalization will silently misbehave.

## 6. Decided

`backend=miles` does **not** also support Miles' own rollout path (rLLM as trainer only). It
would double the config surface for no gain — rLLM's agent/workflow layer is the reason to use
rLLM at all.
