# Design: Config-file-driven `rllm train` / `rllm eval`

- **Status:** Proposed. One deliverable: a single self-contained config file (`.toml`, `.yaml` also accepted) that fully specifies a run, launched as `rllm train run.toml` / `rllm eval run.toml`. Two design forks resolved with a recommendation below (schema = **hybrid**; custom code = **import-path + optional entrypoint hook**); alternatives recorded in *Alternatives considered*.
- **Scope:** Make the config file the positional, self-contained source of truth for a run — carrying the *run definition* (agent, datasets, evaluator, sandbox) that today lives as **code** in each cookbook's `train.py`, plus every config knob that today lives as **Hydra dotlist args** in a `train_*.sh`. CLI flags/dotlist become optional overrides on top. A design goal is **cross-backend portability** — one file where flipping `backend` reuses the common config and only the backend-specific section changes (see [Backend selection & cross-backend portability](#backend-selection--cross-backend-portability)); reaching it fully depends on a schema-normalization workstream this feature motivates. Otherwise no change to the trainer, backends, or config semantics — this is a new front-end over the existing `AgentTrainer` facade.
- **Non-goals:** Replacing Hydra for power users who prefer `python train.py <overrides>` (that path stays). Changing the config schema/tree. A new resolver for agents/datasets/evaluators (we reuse `load_agent` / `load_evaluator` / `DatasetRegistry` / `BenchmarkLoader`).
- **Related:** extends the existing `--config file.yaml` merge in `rllm/cli/train.py` and `rllm/cli/sft.py`; reuses the Hydra-free merge in `build_train_config` and the resolution logic in `_run_train` / `_run_eval`.

## Summary

Today a real training run is a **pair**: a cookbook `train.py` (Python — loads datasets, constructs the `AgentFlow`, reads env vars, calls `AgentTrainer(...).train()`) plus a `train_*.sh` shell script carrying ~40 Hydra dotlist overrides. The quick path (`rllm train <benchmark>`) is a Click command with fixed flags, tinker-only, resolving everything from the catalog. Neither lets you point at *one file* and go.

This design adds a **config-file mode** to `rllm train` / `rllm eval`: `rllm train run.toml`. The file has a small curated **`[run]`** section that captures exactly what the `train.py` *code* does (which agent, which datasets, which evaluator, which sandbox), and everything else **mirrors the existing config tree 1:1** (`[model]`, `[training]`, `[rllm.async_training]`, …) — i.e. your shell-script overrides, as a file. CLI dotlist/flags still apply on top. The same file drives eval, so `rllm eval run.toml` re-runs the exact agent/dataset/sandbox config you trained with.

The whole `terminal-rl` a35b run — `train.py` (110 lines) + `train_fireworks_a35b.sh` (~110 lines of env + Hydra) — collapses to a single ~40-line `run.toml`.

## Motivation

There are two launch surfaces today and they don't overlap:

1. **Power path (every serious run).** `cookbooks/<x>/train.py` under `@hydra.main(config_name="unified")` + a `train_*.sh`. The `.py` holds *code*: dataset loading, `agent_flow = Terminus2Harness(sandbox_backend=…, max_turns=…, enable_summarize=…)`, a dozen `os.environ.get(...)` reads, then `AgentTrainer(...).train()`. The `.sh` holds the config as Hydra dotlist (`rllm/backend=fireworks model.name=… training.group_size=16 rllm.async_training.enable=true …`). Pain: the run definition is split across a `.py` + a `.sh`; config is opaque positional args; env vars silently steer behavior; the three `train_fireworks_{,_a35b,_glm5p2}.sh` scripts are 90% duplicated.

2. **Quick path (`rllm train <benchmark>`).** Click flags + optional `--config file.yaml` merged on top; backend hardcoded to tinker; agent/evaluator/dataset resolved from the catalog + local `dataset.toml` dirs. Good for one-liners, can't express a real MoE-on-Fireworks async run.

We already have most of the machinery: `build_train_config` reproduces the Hydra compose (`base.yaml` + `backend/<backend>.yaml`) **without the Hydra runtime**; `--config` already merges a user file over templates; `load_agent("mod:Class")` / `load_evaluator("mod:Class")` resolve code by import path; `DatasetRegistry` / `BenchmarkLoader` / harbor resolution already back `_run_train`. The gap is only that the file (a) isn't the positional arg, (b) isn't *self-contained* (carries neither the run definition nor the backend), and (c) is YAML-over-flags rather than the source of truth.

## Current state (what we build on)

**Config composition (Hydra).** `unified.yaml` composes `rllm/base.yaml` (backend-agnostic, under the `rllm.*` namespace) + `rllm/backend/<backend>.yaml`. Backend templates use `# @package _global_`: a top-level block (`model`, `training`, `data`, `fireworks_config`, …) *and* a nested `rllm:` block that overrides into the `rllm.*` namespace. There are `${oc.select:…}` / `${rllm.…}` interpolations. `build_train_config` (in `rllm/cli/train.py`) already merges this by hand: load `base.yaml` under `{"rllm": …}`, split the backend template's top-level vs `rllm:` block, merge in order.

**Trainer facade.** `rllm.trainer.AgentTrainer` (the `unified_trainer.py:1052` class) is the public API:

```python
AgentTrainer(
    config,                       # OmegaConf DictConfig
    backend="verl"|"tinker"|"fireworks",
    agent_flow=..., evaluator=..., hooks=...,   # OR workflow_class=...
    train_dataset=..., val_dataset=...,
    sandbox_backend=..., sandbox_concurrency=...,
).train()
```

It already auto-wires `SandboxTaskHooks` + gateway loopback/tunnel when the flow/data needs a sandbox, and dispatches to the per-backend launcher by the `backend` string. **This is the single call the config-file loader ultimately makes** — no trainer changes.

**Resolution logic.** `_run_train` / `_run_eval` in the CLI already resolve: local benchmark dirs (`BenchmarkLoader.is_local_benchmark`), catalog datasets (auto-pull), harbor datasets (`harbor:` prefix, row-wrapping), agents (`load_agent`), evaluators (explicit → dataset.toml `[verifier]` → catalog `reward_fn` → per-task). The config-file `[run]` resolver reuses these — factored into shared helpers.

## Design

### File detection

`rllm train <arg>` / `rllm eval <arg>`: if `<arg>` is an existing regular file whose suffix is `.toml` / `.yaml` / `.yml` → **config-file mode**. Otherwise the current behavior (benchmark name, `harbor:` prefix, or local benchmark *directory*). No collision: benchmarks are names or directories, configs are files.

### Schema (hybrid: curated `[run]` + config-tree mirror)

TOML is primary (the ask); YAML accepted via `OmegaConf.load`. The file has three tiers:

**1. Top level — run identity.**

| key | meaning |
|---|---|
| `backend` | `"tinker"` \| `"fireworks"` \| `"verl"`. Selects the backend template. Default `"tinker"`. |
| `extends` | *(optional)* path to another config file merged **underneath** this one. Lets `a35b`/`9b`/`glm5p2` share a base and override only what differs. |

**2. `[run]` — the run definition (what `train.py` code does today).** The only *new* schema surface: the declarative form of "construct these objects". Grouped into sub-tables so the agent / dataset / sandbox concerns stay separate and scannable.

`[run.agent]`

| key | meaning |
|---|---|
| `name` | built-in catalog name (`terminus2`, `mini-swe-agent`, `react`, `claude-code`, `aider`, `codex`, `opencode`, `oracle`, …), a user-registered name (`~/.rllm/agents.json`), a `harbor:<scaffold>` prefix, **or** a `module:Class` import path for anything unregistered. Bare names are the normal case; the import path is the escape hatch. Resolved by `load_agent` (order: user registry → import path → built-in catalog → entry-point plugin). |
| `[run.agent.args]` | table → constructor kwargs for the flow (`max_turns`, `enable_summarize`, …). Absorbs the per-cookbook `TERMINUS_*` env reads. |
| `evaluator` | *(optional)* registry name / import path; **omit** → per-task verifier from `dataset.toml`/`task.toml` (`SandboxTaskHooks`). |

`[run.dataset]`

| key | meaning |
|---|---|
| `train` / `train_split` | registry name, local benchmark dir, or `harbor:<name>`; split resolved as `_run_train` does. |
| `val` / `val_split` | same; omit `val` to reuse the train tasks for validation. |
| `max_examples` | cap the train set. |

`[run.sandbox]`

| key | meaning |
|---|---|
| `backend` | `docker`\|`local`\|`modal`\|`daytona`\|…; forwarded to the auto-wired `SandboxTaskHooks`. Omit → the `dataset.toml` `default_sandbox` / per-task default. |
| `concurrency` | override the flow's `max_concurrent`. |

Directly under `[run]`: `entrypoint` (escape hatch, `"module:function"` — see [Custom code](#custom-code-import-path--optional-entrypoint-hook)) and `env` (a table of env vars exported before the run — see below).

**Why nested under `[run.*]`, not bare `[agent]`/`[sandbox]`/`[dataset]`.** The config tree already owns those names with *different* meanings, and the tier-3 mirror rule maps a bare section straight onto the tree: `[data]` = batch sizes & sequence lengths (read across every backend), `[agent]` = the *workflow-path* agent (`agent.name`/`max_steps`, resolved via `env_agent_mappings`), `[env]` = the *workflow-path* environment. A bare `[agent]` would merge into `config.agent` — silently ignored by the AgentFlow path but overwriting the workflow default — and a `[dataset]` next to `[data]` reads as a typo. Nesting under `[run.*]` gives the split-out readability while staying collision-free.

> **Resolver note.** Built-in/registered names today instantiate the flow with *no* constructor args (`load_agent` calls `Cls()`), so `[run.agent.args]` requires a small resolver change: instantiate the catalog class *with* those kwargs, so `agent.args` behaves as constructor kwargs uniformly whether the agent was named or imported. (Alternative for flows that only expose knobs via `configure()`: route `args` through `configure()` — but constructor kwargs are the cleaner default.)

**3. Config-tree mirror — everything else, 1:1 with the Hydra tree.** Any section the trainer config already has: `[model]`, `[training]`, `[validation]`, `[data]`, `[fireworks_config]`, `[rollout_engine]`, `[concurrency]`, and the `rllm.*` namespace as `[rllm.trainer]`, `[rllm.algorithm]`, `[rllm.async_training]`, `[rllm.gateway]`, `[rllm.workflow]`, `[rllm.rollout.train]`, `[rllm.rollout.val]`, `[rllm.compact_filtering]`, `[rllm.rejection_sample]`, … These merge directly onto the composed config — no translation, so they never drift from the schema. **For a file you intend to flip across backends, author common knobs in the canonical `rllm.*` namespace (or its normalized aliases) rather than the backend-flavored top-level `model`/`training` sections — see [Backend selection & cross-backend portability](#backend-selection--cross-backend-portability).**

**Convenience the loader applies** (so the file stays DRY vs. today's shell scripts):
- `[data]` is auto-mirrored into `[rllm.data]` (today's scripts write both `data.*` and `rllm.data.*`; `sync_config` keeps parity, but the file should require it once).
- `[run.env]` *(optional)* — a table of environment variables exported before the run, capturing the `export RLLM_HARNESS_RUN_TIMEOUT_S=… / RLLM_SANDBOX_TIMEOUT_S=… / RLLM_SANDBOX_LOG_CAPTURE_DIR=…` lines from the shell scripts. Keeps infra knobs in the file instead of the shell. (Under `[run.*]` for the same reason as above — bare `[env]` is the workflow-path environment.)

### Worked example — the terminal-rl a35b run, as one file

`cookbooks/terminal-rl/train.py` + `train_fireworks_a35b.sh` → `run.toml`:

```toml
backend = "fireworks"

[run.agent]                                            # ← was train.py code
name = "terminus2"                                     # built-in catalog name
# evaluator omitted → per-task sandbox-shell verifier (tests/test.sh)
[run.agent.args]                                       # ← was TERMINUS_* env vars
max_turns        = 75
enable_summarize = false

[run.dataset]
train = "tb-opus-pass"
val   = "terminal-bench@2.0"

[run.sandbox]
backend = "modal"

[run.env]                                              # ← was `export ...` lines
RLLM_HARNESS_RUN_TIMEOUT_S   = "2400"
RLLM_SANDBOX_TIMEOUT_S       = "3200"
RLLM_SANDBOX_LOG_CAPTURE_DIR = "sandbox_logs/qwen3p5-35b-a3b"

[model]
name            = "accounts/fireworks/models/qwen3p5-35b-a3b"
tokenizer_model = "Qwen/Qwen3.5-35B-A3B"
lora_rank       = 32

[training]
group_size    = 16
learning_rate = 2e-5
max_length    = 139264

[data]                                                 # auto-mirrored to [rllm.data]
max_prompt_length   = 122880
max_response_length = 16384
train_batch_size    = 1
val_batch_size      = -1

[fireworks_config]
policy_trainer_shape_id          = "accounts/fireworks/trainingShapes/qwen3p5-35b-a3b-256k-lora"
policy_trainer_replica_count     = 2
rollout_deployment_replica_count = 6

[rllm.async_training]
enable                     = true
mini_batch_size            = 12
fwd_bwd_group_size         = 1
staleness_threshold        = 3.0
trigger_parameter_sync_step = 1
partial_rollout            = true

[rllm.gateway]
port                 = 9090
tunnel               = "https://rllm.ngrok.dev"
num_workers          = 4
cumulative_token_mode = true
renderer_family      = "qwen3.5"

[rllm.workflow]
n_parallel_tasks = 256
raise_on_error   = false

[rllm.algorithm]
adv_estimator          = "grpo"
norm_adv_by_std_in_grpo = true

[rllm.rejection_sample]
filter_uniform_groups = true

[rllm.rollout.train]
temperature = 1.0
top_p       = 0.95

[rllm.rollout.val]
temperature = 1.0
top_p       = 0.95

[rllm.trainer]
total_epochs    = 1
logger          = ["wandb"]
project_name    = "terminal-rl"
experiment_name = "terminal-rl-terminus2-qwen3p5-35b-a3b-fireworks"
val_before_train = false
test_freq       = 100
save_freq       = 10
```

Launch: `rllm train run.toml`. Override for a sweep without editing the file:
`rllm train run.toml training.learning_rate=1e-5 rllm.trainer.experiment_name=lr1e5`.

The `9b` / `glm5p2` variants become tiny files with `extends = "run.toml"` overriding only `model.*` + `fireworks_config.*` — killing the shell-script duplication.

> This example authors common knobs in fireworks' native sections (`[training]`, `[model]`, `[fireworks_config]`) — fine for a single-backend file. For a file you intend to flip across backends, author them in the canonical namespace instead (see [Backend selection & cross-backend portability](#backend-selection--cross-backend-portability)).

### Resolution & precedence

Merge order (lowest → highest; standard OmegaConf merge, matching Hydra intuition and today's `--config < flags` rule):

```
base.yaml + backend/<backend>.yaml          # composed template (build_train_config path)
  ← extends chain (if any), file underneath file
  ← the config file's mirror sections        # [model], [training], [rllm.*], ...
  ← [data] → [rllm.data] auto-mirror
  ← CLI dotlist overrides (key=value)         # `training.learning_rate=1e-5`
  ← CLI named flags (--lr, --experiment, …)   # when explicitly passed
```

`backend` is read first (file, overridable by a `backend=` dotlist / `--backend`) to pick the template. Named flags keep their current defaults but only override when the user actually passed them (so a flag's default never clobbers a file value — implemented by treating unset flags as absent, as `train.py` already does for `--ui`).

### Loading pipeline

New module `rllm/config/run_config.py` (importable outside the CLI too):

```python
def load_run_config(path, *, overrides=None, cli_flags=None) -> tuple[DictConfig, RunSpec]:
    raw   = _parse(path)                     # tomllib for .toml, OmegaConf for .yaml
    chain = _resolve_extends(raw, path)      # merge `extends` bases underneath
    backend = _pick_backend(chain, overrides)
    cfg   = merge_backend_config(backend, user_cfg=chain)   # ← extracted from build_train_config
    cfg   = _mirror_data(cfg)                # [data] → [rllm.data]
    cfg   = _apply_overrides(cfg, overrides, cli_flags)
    run   = RunSpec.from_config(chain)       # the [run] section (+ [env], backend)
    return cfg, run
```

`RunSpec` is a small dataclass of the `[run]` keys. Then a thin driver (shared by train and eval):

```python
cfg, run = load_run_config(path, overrides=dotlist, cli_flags=flags)
_export_env(run.env)                      # run.env  == [run.env]
objs = resolve_run(run, cfg)   # ← extracted from _run_train / _run_eval resolution
AgentTrainer(cfg, backend=run.backend, **objs,
             sandbox_backend=run.sandbox.backend,        # [run.sandbox]
             sandbox_concurrency=run.sandbox.concurrency).train()
```

`resolve_run` returns `{agent_flow, evaluator, train_dataset, val_dataset}` (or `{hooks, ...}`), reusing the exact catalog/local/harbor logic in `_run_train`. **Refactor:** lift that resolution out of `_run_train`/`_run_eval` into shared functions so flag-mode and file-mode share one implementation (they're near-duplicates already).

### Custom code (import-path + optional entrypoint hook)

Most cookbooks are pure "construct this harness with these kwargs + load these datasets" — fully expressible by `[run]` (agent import path + `agent_args`, dataset names, sandbox). `terminal-rl`'s `train.py` is exactly this shape and needs **no** Python file under the new scheme.

For runs that genuinely need Python (dataset built dynamically, a custom evaluator wired at runtime, conditional logic), `[run].entrypoint = "module:function"` is the escape hatch:

```python
# my_cookbook/setup.py
def build(cfg):                       # receives the fully-merged DictConfig
    return {
        "agent_flow":   MyFlow(temperature=cfg.rollout.train.temperature),
        "evaluator":    MyEvaluator(...),
        "train_dataset": make_dataset(...),
        "val_dataset":   make_dataset(...),
    }
```

```toml
[run]
entrypoint = "my_cookbook.setup:build"
```

When `entrypoint` is set, its returned dict supersedes the declarative `agent`/`dataset`/`evaluator` keys (which may then be omitted). This keeps the common case file-only while never boxing out arbitrary setup — the cookbook shrinks from "a `train.py` with a Hydra `main()` + a shell script" to "a `build()` function + a TOML".

### Backend selection & cross-backend portability

`backend` selects both the composed template (`merge_backend_config`) and the launcher (`AgentTrainer(backend=...)`) — all three already dispatch through the facade, so file-mode just lifts the tinker-only hardcode in `rllm/cli/train.py` (verl rides the same facade; its launcher handles Ray init).

**Goal: one file, flip `backend`, common config carries over.** This is already the architecture's intent. `rllm.*` is the canonical backend-agnostic namespace, and each backend ships a `sync_config` (`rllm/trainer/<backend>/utils.py`) whose `_SHARED_KEYS` table fans a common `rllm.*` key out to that backend's native path (verl → `actor_rollout_ref.*` / `data.*` / `trainer.*`; tinker/fireworks → top-level `data.*` / `training.*`), with an explicit `rllm.*` value always winning. So the file authors common config **once in the canonical namespace**, and the loader runs the active backend's `sync_config` to populate its native keys.

**What's portable today vs. the gap** — coverage is uneven:

| knob | canonical `rllm.*` home | ports across backends today |
|---|---|---|
| adv_estimator, loss_fn/agg, kl_beta, eps_clip, warmup, sampling | ✅ `rllm.algorithm.*` / `rllm.rollout.*` | ✅ |
| batch sizes, seq lengths | ✅ `rllm.data.*` | ✅ |
| trainer freqs, epochs, logger, project/exp | ✅ `rllm.trainer.*` | ✅ |
| group size | ✅ `rllm.rollout.n` | ⚠️ verl syncs it; tinker/fireworks interpolate from `training.group_size` |
| **model name / tokenizer** | ❌ `model.name` vs verl `actor_rollout_ref.model.path` | ❌ |
| **learning rate / optimizer** | ❌ `training.learning_rate` vs verl `actor_rollout_ref.actor.optim.lr` (not in `_SHARED_KEYS`) | ❌ |
| **LoRA rank / targets** | ❌ `model.lora_rank` + per-backend target flags | ❌ |

~Half the common surface already ports; the rest (model identity, LR/optimizer, LoRA, group-size) has no single home, so flipping `backend` today means re-stating those in the new backend's native paths.

**Closing the gap** — a schema-normalization workstream this feature motivates; each step is a small, independently-landable addition to the sync tables (same pattern as the ~35 verl entries): promote the un-canonicalized knobs into `rllm.*` and extend each `_SHARED_KEYS` — `rllm.model.{name, tokenizer, lora.*}`, `rllm.optim.{lr, betas, eps, weight_decay, grad_clip}`, and finish `rllm.rollout.n` as the sole group-size home (deprecate `training.group_size`). The **file schema doesn't change** as this lands; knobs just migrate from a `[<backend>]` section into the common block.

**Backend-specific config stays backend-scoped**, read only when that backend is active — no cross-backend meaning: verl resources (`nnodes` / `n_gpus_per_node`) / Megatron / critic / placement; fireworks shapes / replicas / provisioning (`fireworks_config.*`, `fireworks_infra.*`, `concurrency.*`); tinker `num_minibatches` / LoRA-target flags / `tinker_base_url`.

**File shape** — backend-agnostic common block + one section per backend; flip the top line to switch:

```toml
backend = "fireworks"        # ← the only line that changes to switch backend

[model]                      # common (canonical rllm.model.* after normalization)
name = "…/qwen3p5-35b-a3b"
tokenizer_model = "Qwen/Qwen3.5-35B-A3B"
[model.lora]
rank = 32
[optim]                      # common (canonical rllm.optim.*)
lr = 2e-5
[algorithm]                  # common (rllm.algorithm)
adv_estimator = "grpo"
[data]                       # common (rllm.data)
max_prompt_length = 122880

[fireworks]                  # backend-specific — read only when backend="fireworks"
policy_trainer_shape_id = "…/qwen3p5-35b-a3b-256k-lora"
rollout_deployment_replica_count = 6
[tinker]                     # ignored unless backend="tinker"
num_minibatches = 1
[verl]                       # ignored unless backend="verl"
nnodes = 1
n_gpus_per_node = 8
```

The loader routes common sections to the canonical namespace (then `sync_config` fans them native) and the active `[<backend>]` section to that backend's native tree; inactive backend sections are ignored but kept for one-flip switching. **MVP vs. north star:** file-mode ships *now* against today's uneven canonicalization — portable for the already-shared half, with the rest expressed in the `[<backend>]` sections — and becomes fully portable as the normalization lands incrementally.

### Eval path

The **same file** drives eval. `rllm eval run.toml` reads `[run]` (agent, evaluator, dataset/split, sandbox) + the model source, and an optional `[eval]` section for eval-only knobs:

```toml
[eval]
model       = "accounts/fireworks/models/qwen3p5-35b-a3b"  # or base_url / provider from `rllm setup`
split       = "default"
concurrency = 64
attempts    = 1
# max_examples, task_indices, sampling, snapshot, warm_queue_size, output — as CLI flags today
```

`_run_eval` already does all the resolution; file-mode just populates its arguments from `[run]` + `[eval]` instead of Click flags. Model source precedence: `[eval].base_url` (direct) → `[eval].model`/`[eval].provider` → `rllm setup` config, mirroring `eval_cmd` today. **Payoff:** train and eval read one file, so eval is guaranteed to use the same agent/sandbox/dataset config as training — no drift between a `train.py` and a separate eval invocation.

## Implementation plan

Phased, each independently landable:

1. **Extract shared helpers (no behavior change).**
   - `merge_backend_config(backend, user_cfg)` ← the compose logic in `build_train_config` (`rllm/cli/train.py:32`).
   - `resolve_run_objects(...)` ← the agent/evaluator/dataset resolution in `_run_train` (`:146`) and `_run_eval` (`:117`). Land as refactor; existing flag paths call the extracted fns.
2. **`rllm/config/run_config.py`.** `RunSpec`, `load_run_config`, `_parse` (tomllib/YAML), `_resolve_extends`, `_mirror_data`, `_apply_overrides`, `_export_env`. Unit-tested against the a35b example → expected merged `DictConfig`.
3. **Wire `train_cmd`.** File detection on the positional arg; on match, call the file driver (dotlist from remaining args, named flags as overrides). Non-file arg → unchanged.
4. **Wire `eval_cmd`.** Same detection; `[eval]` + `[run]` → `_run_eval` args.
5. **Entrypoint hook.** `[run].entrypoint` import + `build(cfg)` contract; supersede declarative keys.
6. **Docs + reference migration.** Convert `cookbooks/terminal-rl` to `run.toml` (+ `run_eval.toml` or shared) as the worked example; keep `train.py`/`.sh` temporarily for A/B. Doc page under `rllm-docs`.

**Files touched:** `rllm/config/run_config.py` (new), `rllm/cli/train.py` (detect + driver + extract), `rllm/cli/eval.py` (detect + driver + extract), `rllm/cli/_run_resolve.py` (new, shared resolution), tests under `tests/`, one cookbook + docs. No trainer/backend changes.

**Dependencies:** `tomllib` (stdlib ≥3.11; repo is Python ≤3.12 — fine). YAML already supported by OmegaConf.

## Migration

- Cookbook shell + `train.py` pairs → one `run.toml` (pure-declarative cookbooks) or `run.toml` + a small `build()` (custom ones). Old `python train.py <hydra args>` keeps working — the file mode is additive.
- The near-duplicate `train_fireworks_{,_a35b,_glm5p2}.sh` collapse to a base `run.toml` + tiny `extends` variants.
- `rllm train <benchmark> --flags` (quick path) is unchanged.

## Alternatives considered

**Schema shape.**
- **Strict mirror** (every section is a Hydra node, no curated layer): zero new schema to maintain, but agent/dataset/evaluator/sandbox have *no home in the config tree today*, so they'd be bolted on as ad-hoc raw keys anyway — you invent a `[run]`-equivalent regardless, with less clarity. Rejected.
- **Curated/friendly flat schema** (friendly names, loader expands to nested): nicest for newcomers, but it's a translation layer that must track the schema as it evolves, and every advanced knob needs an `[overrides]` escape hatch — which is most of a real run. High maintenance for marginal gain over hybrid. Rejected.
- **Hybrid (chosen):** curated `[run]` (the genuinely-new run-definition surface) + 1:1 mirror for everything else. The mirror never drifts (it *is* the tree); the curated part is tiny and stable. Lowest total maintenance, self-contained, and the mirror keys already match muscle memory from the shell-script dotlists.

**Custom code.**
- **Import-path only** (no `entrypoint`): smaller surface, but any dynamic dataset/evaluator forces a separate `.py` you run directly — so those runs can't use `rllm train file.toml` at all, undercutting the feature for exactly the complex runs that most want one entry point. Rejected as the *only* mechanism.
- **TOML = config only, keep `train.py`:** minimal change, but doesn't deliver a real `rllm train file.toml` for the power path — the run definition stays split across `.py` + file. Rejected.
- **Import-path + optional entrypoint hook (chosen):** declarative for the common case (no `.py`), `build(cfg)` escape hatch for arbitrary setup. Covers both ends of the spectrum with one loader.

## Open questions

- **`extends` semantics:** single base or a list? Relative to the file or CWD? (Lean: single path or list, resolved relative to the file.)
- **Named-flag surface in file mode:** which Click flags remain meaningful as overrides (e.g. `--experiment`, `--max-examples`, `--sandbox-backend`) vs. file-only? (Lean: keep the run-shaping flags as overrides; drop redundant ones.)
- **One file for both, or `[train]`/`[eval]` split?** A shared `[run]` + separate `[eval]` reads cleanest and keeps train/eval parity; confirm we don't want fully separate files.
- **Validation & errors:** unknown top-level keys → hard error (typo protection) vs. warn? Schema-validate `[run]` against `RunSpec`; mirror sections pass through to OmegaConf (which already errors on struct violations at merge if we enable struct mode).
- **Backend-section grouping:** clean aliases (`[fireworks]`/`[tinker]`/`[verl]`, `[optim]`, `[model.lora]`) that need a thin routing map, vs. native section names (`[fireworks_config]`, `[actor_rollout_ref]`, …) that mirror 1:1 but leak each backend's native layout into the file? (Lean: alias sugar for the handful of backend + normalized-common blocks; 1:1 mirror for everything else.)
- **Normalization sequencing:** ship file-mode against today's uneven `_SHARED_KEYS` first (backend sections cover the un-shared knobs), then promote model/optimizer/LoRA/group-size into `rllm.*` incrementally — or block file-mode on the normalization landing? (Lean: ship first, normalize incrementally; the file schema is stable across it.)
