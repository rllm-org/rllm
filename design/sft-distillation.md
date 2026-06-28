# Design: Eval → Curate → SFT in the rLLM CLI

- **Status:** Proposed. Milestone 1 (curation engine) landing first; milestones 2–5 staged behind it.
- **Scope:** `rllm eval` (already saves trajectories), a new `rllm dataset from-eval` curation command, and a unified `rllm sft` over verl/tinker/fireworks backends.
- **Related:** mirrors the RL backend abstraction (`AgentTrainer` / `TrainerLauncher` / `UnifiedTrainer` in `rllm/trainer/unified_trainer.py`).

## Summary

We want a seamless loop: run an eval with `k` samples per task, **curate** the
resulting trajectories by aggregate metrics (avg@k / pass@k), turn the survivors
into an SFT dataset, and train on it — all from the CLI:

```bash
# 1. Eval with k samples per task — already saves full trajectories today
rllm eval math500 --model Qwen/Qwen2.5-7B-Instruct --attempts 8

# 2. NEW: curate trajectories by aggregate metrics → a registered SFT dataset
rllm dataset from-eval math500_Qwen2.5-7B-Instruct_20260620_141500 \
    --name math500-rft --filter "0 < avg < 1" --select correct --max-per-task 2

# 3. Train SFT on it
rllm sft math500-rft --model Qwen/Qwen2.5-7B-Instruct --backend tinker --epochs 3
```

~80% of this already exists. Eval persists full trajectories and computes
pass@k; a dataset registry writes both parquet flavors; an SFT trainer exists for
verl and tinker. The two gaps are: (a) there is no **curation** step that
aggregates the `k` attempts per task, filters, selects trajectories, and registers
an SFT dataset; and (b) `rllm sft` is written but orphaned, tinker-only, and its
config plumbing leaks into the CLI. This doc fills both.

## Motivation

The "RL eval → SFT" loop (rejection sampling / STaR / RFT) is a staple: sample
the model `k` times per task, keep the trajectories from tasks the model can
*sometimes* solve, and fine-tune on the successful ones. Today a user must
hand-write a script that loads saved episodes, recomputes per-task aggregates,
filters, reshapes to `{"messages": [...]}`, registers a dataset, and then invokes
a broken experimental SFT command. Every piece exists; nothing is wired.

## What already exists (build on, don't rebuild)

| Capability | Where | State |
|---|---|---|
| Eval saves full trajectories per task | `rllm eval --save-episodes` (default) → `~/.rllm/eval_results/<run_id>/episodes/episode_NNNNNN_<task_id>.json` | works |
| `k` samples per task (avg@k / pass@k source) | `rllm eval --attempts N`; runner expands tasks, stamps `eval_idx`; `results.json` carries per-rollout `items` (`idx`, `attempt`, `reward`, `is_correct`, `signals`) and aggregate `pass_at` | works |
| Unbiased pass@k | `rllm/eval/results.py:_pass_at_k` | works (reused by curation) |
| Dataset registry (parquet + verl parquet) | `DatasetRegistry.register_dataset(name, data, split, ...)`; `rllm dataset list/info/inspect` | works; SFT format = `{"messages": [...]}` |
| SFT trainer (verl + tinker) | `AgentSFTTrainer._train_verl` / `_train_tinker`, `RLLMSFTDataset` | works but ad-hoc; no shared contract |
| `rllm sft` CLI | `rllm/experimental/cli/sft.py` | orphaned: not registered; broken imports; backend hardcoded to tinker; `build_sft_config` leaks tinker's schema into the CLI |

## Part A — the curation engine

### Data flow

```
eval run dir/                          rllm dataset from-eval
 ├─ results.json  ──(cheap)──►  group items by stable task_id (pooled across runs)
 │   items[]: idx,attempt,             compute per-task aggregates: avg, pass@k, best, n, n_correct
 │   reward,is_correct,signals         │
 │                              ── task-level --filter ──►  surviving tasks
 │                                      │
 │                              ── per-task --select ───►  chosen (task, attempt) set
 │                                      │
 └─ episodes/episode_NNNNNN_*  ─(load only chosen)─► trajectories[0].steps[-1].chat_completions
                                        │
                                        ▼
                       [{"messages":[...], source_run, task_id, attempt, score}]
                                        │
                                        └─► DatasetRegistry.register_dataset(name)  ── writes .parquet AND _verl.parquet
```

Filtering and reward/`is_correct`-based selection run off the lightweight
`results.json` `items`, so only the chosen episode JSONs are deserialized.

### Mapping items to episodes

The runner expands each task into `attempts` adjacent rollouts; for a results
item `(task_idx, attempt)` the expanded index is `eval_idx = task_idx * attempts +
attempt` (and `eval_idx == idx` when `attempts == 1`). Episode files are named
`episode_{eval_idx:06d}_{task_id}.json`, so the engine builds an `eval_idx →
(path, task_id)` index once per run by listing the `episodes/` directory — no need
to open the episodes for the filter pass.

### Grouping & multi-run pooling

Groups are keyed on the stable `task_id` recovered from the episode filename
(falling back to `<run_id>:t<idx>` when absent). Passing several run dirs
therefore **pools attempts for the same task across runs** — re-running eval
accumulates samples for a sharper `avg` / `pass@k`. `Task.id` is unique per
dataset by construction, so single-run grouping is exact.

### The `--filter` DSL

A small boolean expression evaluated per task. `<name>@<k>` tokens are rewritten
to a whitelisted accessor (`pass@4` → `_at("pass", 4)`) before parsing, then the
AST is validated against a node whitelist (comparisons, `and`/`or`/`not`, numeric
literals, and the single `_at` call — no attributes, no other calls). Bindings:

| Name | k-dependent? | Definition |
|---|---|---|
| `avg` (`avg@k` accepted, k ignored) | no | mean of `--metric` over attempts. **avg@k is mathematically k-invariant** (E[mean of a random k-subset] = the overall mean), so the metric for difficulty bands is just `avg`. |
| `pass@k` | **yes** | unbiased pass@k (reuses `_pass_at_k`) — budget-aware "solvable within k tries". |
| `best` / `worst` | no | observed max / min of `--metric`. |
| `solved` | no | `n_correct > 0` (≥1 success). |
| `n`, `n_correct` | — | attempt count, success count. |

```bash
--filter "solved"               # default: any success exists (pure rejection sampling)
--filter "0 < avg < 1"          # difficulty band: drop trivial & impossible
--filter "pass@4 >= 0.5"        # solvable ≥50% of the time within 4 tries
--filter "best == 1 and avg < 0.5"   # solvable but usually fails → high learning value
```

`--metric` chooses what is averaged: `is_correct` (default, 0/1), `reward`
(continuous), or any signal name (e.g. `accuracy`).

### Trajectory selection (which attempts per surviving task)

- `correct` (default) — all *passing* attempts (`is_correct`, or `score >=
  --min-reward` when set). Rejection sampling / RFT.
- `best` — single highest-scoring passing attempt.
- `best-n` — top `--max-per-task` passing attempts by score.
- `shortest` — among passing, the shortest by emitted content length (the one
  selector that peeks at episodes).
- `all` — every attempt (distillation, not rejection sampling).

`--max-per-task` caps trajectories per task; `--dedup` drops identical
assistant solutions; each row carries provenance (`source_run`, `task_id`,
`attempt`, `score`).

### `messages` extraction & robustness

Primary source is `trajectories[0].steps[-1].chat_completions` (matches eval's
own scoring at `runner.py`). Fallback: walk steps backward to the last non-empty
`chat_completions`; if none yields ≥2 clean turns, **skip with a counted
warning** rather than emit an empty row. `--trajectory <name>` selects a named
trajectory for multi-agent flows (default: first).

### Module layout

- `rllm/eval/filter_dsl.py` — `compile_filter(expr) -> CompiledFilter` (the
  `@k` rewrite + AST-whitelist evaluator).
- `rllm/eval/curation.py` — `curate(run_dirs, CurationConfig) -> (rows,
  CurationStats)`: load runs, build `AttemptGroup`s, filter, select, lazy-load
  episodes, emit `messages` rows. Absorbs the logic of the (zero-caller, now
  removed) `AgentSFTTrainer.process_trajectories`.
- `rllm/cli/dataset.py` — a thin `from-eval` subcommand over `curate()`, plus
  `--dry-run` (print survivor counts, write nothing) for threshold iteration.

## Part B — a unified SFT trainer

### What RL's "unified" design actually is

RL unifies backends with a 3-layer cake, and the cross-backend seam is the
**dispatcher + launcher**, not a shared loop:

| Layer | Class | Job |
|---|---|---|
| Dispatcher (user-facing) | `AgentTrainer(backend=...)` — "delegates to the corresponding launcher" | normalize config, resolve data, pick backend |
| Launcher | `TrainerLauncher` → `VerlTrainerLauncher` (Ray actor) / `FireworksTrainerLauncher` (in-process) | per-backend environment/topology setup |
| Shared loop | `UnifiedTrainer` + `BackendProtocol` step methods | the training loop — but verl runs largely in its own Ray actor; the shared loop meaningfully spans only the tinker family |

### Why SFT mirrors the dispatcher/launcher, not the loop

For SFT there is **no shareable per-step primitive**: verl SFT is a monolithic
FSDP loop inside a `torchrun` process group (`verl.trainer.sft_trainer.SFTTrainer.fit()`);
tinker SFT is an async loop that pipelines `fwd_bwd`/`optim_step` futures. They
have nothing to factor into shared `forward_backward()`/`optim_step()` calls
without forking vendor internals. So we **unify at the spec + launcher level and
let each backend own its `fit()`** — the honest analogue of how RL already treats
verl.

This also fixes the concrete uncleanliness today: `build_sft_config` lives in the
CLI and hardcodes tinker's config tree, and `AgentSFTTrainer` carries two
unrelated `_train_verl`/`_train_tinker` methods with no contract.

### Shape (clean break — old SFT code removed; the dispatcher keeps the name `AgentSFTTrainer`)

```python
# rllm/trainer/sft/spec.py — backend-AGNOSTIC; what the CLI/curation fills
@dataclass
class SFTSpec:
    model: str
    train_dataset: Dataset
    val_dataset: Dataset | None = None
    lr: float = 1e-5
    lr_schedule: str = "constant"
    epochs: int = 1
    batch_size: int = 32
    max_length: int = 2048
    tokenize_method: str = "cumulative"   # cumulative | stepwise | hf_template
    lora_rank: int = 32
    save_freq: int = 20
    val_freq: int = 10
    project: str = "rllm-sft"
    experiment: str | None = None
    output_dir: str | None = None
    overrides: dict | None = None          # deep-merged into the backend config

# rllm/trainer/sft/backend.py — the contract; each backend owns its loop
class SFTBackend(ABC):
    name: str
    requires_distributed: bool = False     # verl=True (torchrun), tinker/fireworks=False
    def __init__(self, spec: SFTSpec): ...
    @abstractmethod
    def validate_spec(self) -> None: ...
    @abstractmethod
    def build_config(self) -> DictConfig: ...   # SFTSpec → backend-NATIVE schema (lives here, not the CLI)
    @abstractmethod
    def prepare_data(self) -> None: ...          # verl: spec→data.train_files=_verl.parquet ; tinker: Dataset objs
    @abstractmethod
    def fit(self) -> None: ...                    # run the whole loop (delegates to vendor)
    @property
    @abstractmethod
    def checkpoint_dir(self) -> str: ...

# rllm/trainer/agent_sft_trainer.py — the dispatcher (≈ RL AgentTrainer), file fully rewritten
class AgentSFTTrainer:
    _BACKENDS = {"verl": VerlSFTBackend, "tinker": TinkerSFTBackend, "fireworks": FireworksSFTBackend}
    def __init__(self, spec: SFTSpec, backend: str = "verl"):
        self.spec, self.backend_name = spec, backend
    def train(self):
        be = self._BACKENDS[self.backend_name](self.spec)
        be.validate_spec(); be.prepare_data()
        if be.requires_distributed and not _inside_torchrun():
            self._launch_torchrun(be)    # rllm sft --backend verl --gpus N → torchrun -m …verl_entry
        else:
            be.fit()                      # tinker/fireworks in-process; verl when already inside torchrun
```

`SFTSpec` is the only input — no legacy `(config, train_dataset, val_dataset)`
signature, no passthrough. The torchrun entry (`rllm/trainer/sft/verl_entry.py`)
sets `RLLM_SFT_IN_TORCHRUN=1` and re-enters `train()`, which then calls
`be.fit()` directly inside the process group. (verl SFT uses torchrun/FSDP, not
Ray — the one way it differs from RL-verl.)

Concrete backends wrap existing code: `TinkerSFTBackend` promotes the loop from
`deprecated/tinker_sft_trainer.py`; `VerlSFTBackend` wraps the verl `SFTTrainer`
body from today's `_train_verl`; `FireworksSFTBackend(TinkerSFTBackend)` overrides
only the client bits (mirroring `FireworksBackend(TinkerBackend)`).

### How the CLI maps

`rllm sft` flags → `SFTSpec` → `AgentSFTTrainer(spec, backend).train()`. The CLI
references no backend config keys. Curation's dual-parquet output feeds either
backend: `prepare_data()` picks `.parquet` (tinker Dataset) or `_verl.parquet`
(verl `data.train_files`).

## File manifest

**Add**
- `rllm/eval/curation.py`, `rllm/eval/filter_dsl.py`
- `rllm/cli/dataset.py` → `from-eval` subcommand
- `rllm/trainer/sft/{spec,backend,tinker_backend,verl_backend,verl_entry}.py`
- `rllm/trainer/sft/config/{tinker,verl}.yaml` (the existing `tinker_sft_trainer.yaml` / `agent_sft_trainer.yaml` move here)
- `rllm/cli/sft.py` (speaks `SFTSpec`; registered in `main.py:_LazyGroup._COMMANDS`)

**Delete**
- old body of `rllm/trainer/agent_sft_trainer.py` (`_train_verl`, `_train_tinker`, `process_trajectories`) → rewritten as the dispatcher
- `rllm/trainer/deprecated/tinker_sft_trainer.py` + `tinker_sft_dataset.py` (logic migrates into `TinkerSFTBackend`) and the `TinkerSFTTrainer` re-export shims in `deprecated/__init__.py` / `tinker/__init__.py`
- `rllm/experimental/cli/sft.py` (incl. `build_sft_config`) and `rllm/experimental/config/sft/base.yaml`

**Keep**
- `rllm/trainer/verl/sft_dataset.py` (`RLLMSFTDataset`) — used by `VerlSFTBackend`

**Update (only external call site)**
- `examples/archive/sft_tinker/train_norobots_tinker.py` → new `AgentSFTTrainer(spec=SFTSpec(...), backend="tinker")` API

## Milestones

1. **Curation engine** — `rllm/eval/curation.py` + `filter_dsl.py` (pure, unit-testable against a saved run dir, no GPU). *Done.*
2. **`rllm dataset from-eval`** + `--dry-run`. *Done.*
3. **SFT abstraction** — `SFTSpec` + `SFTBackend` + `AgentSFTTrainer` dispatcher + `TinkerSFTBackend`; new `rllm/cli/sft.py` registered; delete old SFT code. → loop works on tinker. *Done.*
4. **`FireworksSFTBackend`** — subclasses `TinkerSFTBackend`, reuses the shared `build_sft_data` pipeline + `build_config`, overrides `fit()` with a synchronous pipelined loop (`build_service_client` → `create_training_client` → `ReconnectableClient` → `TrainingCheckpoints`). Hosted/managed like tinker (`requires_distributed=False`), so no launcher needed. *Done.*
5. **`VerlSFTBackend`** + `verl_entry` torchrun launcher + `--gpus`. → `--backend verl` validated multi-GPU.
6. Polish (`shortest`, multi-run pooling tests, `--val-run`); optional fused `rllm distill`.

> The `messages` data pipeline is shared across managed backends via
> `rllm.trainer.sft.tinker_backend.build_sft_data` (tinker-cookbook renderers →
> tinker Datums). Fireworks reuses it wholesale; only client creation and
> checkpointing differ. The default model for managed SFT is `Qwen/Qwen3.5-4B`.

## Open questions

- **Default `--filter`.** Proposed `solved` (≥1 success — never yields empty SFT
  targets from a kept task). Difficulty-band `0 < avg < 1` is opt-in.
- **`best@k`/`worst@k`.** v1 binds `best`/`worst` to *observed* extremes;
  expected max/min over random k-subsets is a later refinement.
