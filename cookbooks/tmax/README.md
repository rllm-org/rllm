# Tmax

Reproduce **[Tmax](https://arxiv.org/abs/2606.23321)** — Ai2's "simple recipe
for terminal agents" (Ivison et al., 2026) — inside rLLM. This cookbook ports
Ai2's published **~14.6K-task `tmax-15k` corpus** as a first-class rLLM dataset
and launches RL on **`Qwen/Qwen3.5-9B`** with hyperparameters tracking their
official **DPPO** run, evaluating on [Terminal-Bench](https://www.tbench.ai).

It's the sibling of [`cookbooks/terminal-rl`](../terminal-rl/README.md): same
machinery (a terminal harness running *inside* each task's sandbox, scored by
the task's own in-sandbox verifier), but the **data is Ai2's** and the **recipe
targets their 9B result**.

Tmax-9B reaches **~27.2% on Terminal-Bench 2.0** (daytona), about **+6 points**
over the `Qwen3.5-9B` base under the same harness, with the best checkpoint at
**step ~200**.

## What "porting their data" means here

Ai2 publishes the corpus on Hugging Face in two complementary forms. We **join
them by `task_id`** to build the native rLLM dataset:

| Form | What it provides | Used here? |
|---|---|---|
| [`allenai/TMax-15K`](https://huggingface.co/datasets/allenai/TMax-15K) | Raw corpus: `description` (task prompt) + **`test_final_state`** (the programmatic pytest verifier — pass/fail is the reward) + `container_def` + `truth`. | **verifier + prompt** |
| [`allenai/tmax-15k-open-instruct`](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct) | Training-ready: per-task **prebuilt image** `env_config.image` = `hamishi740/swerl-tmax-v3:<digest>` (so we pull a ready env instead of building `container_def`). | **prebuilt image** |
| [`tmax/TMax-15K-Harbor`](https://www.harborframework.com/docs/datasets) | Ai2's README says the corpus is *also* on the Harbor registry — but it is **not present in the public Harbor package registry** (`harbor` `list_datasets()` returns 80 datasets, none tmax; harbor 0.13–0.15 all query the same registry). | ✗ not reachable |

The native dataset **`tmax-15k`** is a catalog entry in
[`rllm/registry/datasets.json`](../../rllm/registry/datasets.json) backed by a
builder, [`rllm/data/tmax_builder.py`](../../rllm/data/tmax_builder.py)
(the same pattern as `r2egym` / `swesmith`). `rllm dataset pull tmax-15k`
downloads both HF datasets, joins them, and materializes one Harbor task dir per
task: `FROM` the prebuilt image, `instruction.md` from `description`, and a
`tests/test.sh` that runs `test_final_state.py` under pytest inside the image —
**reward = 1.0 iff it passes**, written to `/tmp/rllm/reward.json` (rLLM's
`script_evaluator` reads that). This reproduces Tmax's outcome-only
`verification_reward` without depending on the missing Harbor registry entry.

> **Docker Hub note:** the per-task images live under `hamishi740/swerl-tmax-v3`.
> Pulling all ~14.6K at training scale wants a Docker Hub business account (their
> README says the same); small subsets pull fine. `--train-limit N` caps what
> *training* uses, but the builder still downloads both full HF datasets (≈200 MB)
> to perform the join.

> **Status:** the builder's join + materialization are validated offline; the
> reward path (`test_final_state.py` pass/fail) follows Ai2's documented
> verifier semantics and rLLM's proven `r2egym` verifier structure, but running
> it end-to-end needs Docker Hub access to the images — confirm on a small subset
> first (Tier 3 below).

## Architecture

```
AgentTrainer.train()
  │
  ├── for each tmax-15k task: launch a sandbox (Modal / Daytona / Docker)
  │       │
  │       └── terminal harness runs IN the sandbox
  │             │   (multi-turn bash/tmux loop; each LLM call → gateway)
  │             │
  │             └── rLLM gateway routes to the trainer-hosted policy,
  │                  capturing the full trajectory.
  │
  └── verifier: tests/test.sh runs test_final_state.py (pytest) in the sandbox
        │   writes {"reward": 1.0/0.0} to /tmp/rllm/reward.json
        │
        └──  →  RL reward signal (Tmax's outcome-only verification_reward)
```

Like terminal-rl, this ships **no custom AgentFlow and no custom evaluator** —
the harness owns the action loop, the gateway owns the trajectory, the
in-sandbox verifier owns the reward.

## Installation

```bash
# full fine-tune (faithful reproduction):
uv pip install -e ".[verl,harbor]"
bash scripts/install_megatron.sh <cu128|cu129|...>

# or an accessible LoRA backend:
uv pip install -e ".[fireworks,harbor]"     # managed
uv pip install -e ".[tinker,harbor]"        # single-machine

# then install this cookbook (registers prepare_data / train):
uv pip install --no-deps -e cookbooks/tmax
```

The `harbor` extra lets the CLI resolve `harbor:` dataset names and ships the
terminal agent code (Terminus-2 / mini-swe-agent install into each task sandbox
on first run — no host-side agent install needed).

## Datasets

```bash
python cookbooks/tmax/prepare_data.py
# or a fast smoke run on a 50-task subset:
python cookbooks/tmax/prepare_data.py --train-limit 50
```

| Dataset | Role | Source | Verifier |
|---|---|---|---|
| `tmax-15k` | train (~14.6K) | HF builder (`allenai/TMax-15K` + `tmax-15k-open-instruct`) | `tests/test.sh` runs `test_final_state.py` (pytest) → `/tmp/rllm/reward.json` |
| `terminal-bench@2.0` | eval (89) | `harbor:terminal-bench@2.0` | in-sandbox `tests/test.sh` → `/logs/verifier/reward.txt` |

`tmax-15k` is a real catalog entry, so you can also pull it standalone (or
materialize a subset directly via the builder for a quick check):

```bash
rllm dataset pull tmax-15k
# or a small offline materialization (no training):
python -m rllm.data.tmax_builder --out-dir /tmp/tmax-smoke --limit 5
```

`TB_EVAL_VERSION` selects the Terminal-Bench eval version (default `2.0`, the
number Tmax reports). The registry publishes `2.0` today; flip to `2.1` once it
lands — `prepare_data.py` and `train.py` both read it so the names stay in sync.

## Training

### verl (full fine-tuning — the faithful reproduction)

```bash
bash cookbooks/tmax/train_verl.sh
```

Full-parameter RL on `Qwen/Qwen3.5-9B` with the Tmax hyperparameters (below).
This is the only backend that matches their **full-parameter DPPO** run at
9B / 65K context / group 32. It is heavy — see **Hardware**.

### Fireworks (managed LoRA) / Tinker (single-machine LoRA)

```bash
export FIREWORKS_API_KEY=...
bash cookbooks/tmax/train_fireworks.sh     # managed trainer + deployment
bash cookbooks/tmax/train_tinker.sh        # single machine
```

`train_fireworks.sh` is aligned **field-for-field** with Tmax's DPPO config —
67584 context (= `pack_length`: 51200 cumulative prompt + 16384 per-turn
response); in fully-async mode prompts/step = `async_training.mini_batch_size=8`
(= `num_unique_prompts_rollout`; `data.train_batch_size` is ignored — async
forces the dataloader batch to 1) × group 32 = 256 rollouts/step; `async_steps=4`
→ `staleness_threshold=3.0`; ~500 steps (`total_batches`, = `total_episodes
128000 / 256`), seed 42, save_freq 20, constant LR, no KL, centered advantages,
64-turn cap, temp 1.0. The **only intended differences are what LoRA + the
backend force**:

- **LoRA-32**, not full-parameter DPPO → higher LR `2e-5` (the paper's `1e-6`
  barely moves a rank-32 adapter).
- **GRPO + PPO clip** instead of DPPO's TV trust region (rLLM has no DPPO); the
  centered-advantage / no-KL / outcome-only parts are matched.
- `async_steps=4` maps only approximately to `async_training.staleness_threshold`;
  `lm_head_fp32` / Liger are backend-internal.

Scale **down** for cost: `ROLLOUT_REPLICAS=2 GROUP_SIZE=8 bash
train_fireworks.sh rllm.async_training.mini_batch_size=2`. `train_tinker.sh` is
the single-machine LoRA sibling.

### Reproduction recipe — Tmax's DPPO config → rLLM

Pulled verbatim from Ai2's
`training/open-instruct/scripts/tmax/RL/qwen35_9b.sh`
(`open_instruct/grpo_fast.py`):

| Tmax (open-instruct) | Value | rLLM equivalent (`train_verl.sh`) |
|---|---|---|
| `model_name_or_path` | `Qwen3.5-9B` | `Qwen/Qwen3.5-9B`, full FT (no LoRA) |
| `num_unique_prompts_rollout` | 8 | `data.train_batch_size=8` |
| `num_samples_per_prompt_rollout` (group) | 32 | `actor_rollout_ref.rollout.n=32` |
| `response_length` (episode) | 65536 | per-turn windows summing to ~64K |
| `per_turn_max_tokens` | 16384 | `data.max_response_length=16384` |
| `max_steps` (tool calls/episode) | 64 | `TERMINUS_MAX_TURNS=64` |
| `learning_rate` | 1e-6 | `actor.optim.lr=1e-6` |
| `lr_scheduler_type` | constant | verl default (`warmup_steps_ratio=0.0`) |
| `beta` (KL) | 0.0 | `rllm.algorithm.kl_beta=0.0` (→ `use_kl_loss=False`) |
| `temperature` | 1.0 | `rollout.temperature=1.0` |
| `advantage_normalization_type` | centered | `rllm.algorithm.norm_adv_by_std_in_grpo=false` |
| `loss_fn` | dppo (tv, thr 0.1) | `rllm.algorithm.adv_estimator=grpo` + PPO clip ⚠️ |
| `verification_reward` | 1.0 | per-task `tests/test.sh` → 1.0/0.0 |
| `seed` | 42 | `rllm.data.seed=42` |
| `num_epochs` | 1 | `trainer.total_epochs=1` |
| `deepspeed_stage` | 3 (full param) | FSDP + param/optimizer offload |

## Fidelity vs. the paper

Honest about the gaps. Reproduce the recipe; expect to be close, not identical.

- **DPPO ⚠️.** rLLM does not implement Tmax's exact DPPO loss (a total-variation
  trust region with divergence threshold 0.1). We map it to **GRPO with
  PPO-style ratio clipping** — the practical analog (outcome-only advantages, no
  learned reward model, on-policy clip as the trust region). The **`centered`
  advantage normalization is reproduced** (`norm_adv_by_std_in_grpo=false`:
  mean-subtract without std division). `lm_head_fp32` and `use_liger_grpo_loss`
  from their script are backend-internal and not exposed.
- **Harness.** Tmax trained with their **Vanillux2** agent (a mini-SWE-agent-
  derived bash-tool harness: submit marker, format-error recovery, output
  truncation). This cookbook defaults to **`terminus2`** (rLLM's proven training
  harness, and the agent the `TMax-15K-Harbor` dataset is documented to run with).
  For the closest match to Vanillux2, set `TMAX_HARNESS=mini-swe-agent`. Different
  scaffolds → different prompts/rollouts, so absolute numbers will shift.
- **Full FT vs LoRA.** Only `train_verl.sh` does full-parameter training.
  `train_fireworks.sh` / `train_tinker.sh` train LoRA adapters (see deviations
  above).
- **Hardware / topology.** Tmax used **8× H100 nodes** (2 for training with
  `sequence_parallel_size 4`, 6 for vLLM inference). rLLM's verl path colocates
  rollout + training (`hybrid_engine`) — a different topology. Full FT of a 9B at
  65K context + group 32 wants ≥2 nodes and/or Ulysses sequence parallelism
  (append `actor_rollout_ref.actor.ulysses_sequence_parallel_size=4`). Set
  `NNODES` / `GPUS_PER_NODE` for `train_verl.sh`.

### Expected results (from the [Tmax-9B model card](https://huggingface.co/allenai/tmax-9b))

| Model | TB Lite | TB 2.1 | TB 2.0 (daytona) |
|---|---|---|---|
| Qwen 3.5 9B (base) | 41.9 | 16.1 | 21.1 |
| **Tmax 9B** | **57.2** | **28.8** | **27.2** |

Best checkpoint at **step ~200**. The optional ECHO variant
(`rllm.algorithm.adv_estimator=echo`) adds free dense supervision from the
terminal output and is a natural fit for a hard, failure-heavy benchmark — see
[terminal-rl](../terminal-rl/README.md#echo-train-on-environment-feedback).

## Evaluation (no training)

```bash
# matches Tmax's reported "TB 2.0 (daytona)" column
rllm eval harbor:terminal-bench@2.0 \
    --agent terminus2 --sandbox-backend daytona \
    --max-tokens 16384 --temperature 1.0 \
    --max-examples 20
```

Per-task results land in `~/.rllm/eval_results/`; `rllm view` opens the
trajectory UI. See the
[Terminal-Bench eval cookbook](../../docs/cookbooks/terminal_bench.mdx) for full
benchmark runs (pass@k, sandbox lifetimes).

## Sandbox backend

Training uses rLLM's own `SandboxedAgentFlow` path (`AgentFlowEngine`). Pick a
backend via `TERMINAL_SANDBOX_BACKEND`:

| Backend | Setup | Notes |
|---|---|---|
| `modal` | `pip install modal` + `modal token new` | Default — per-task billing, scales to many parallel sandboxes. |
| `daytona` | `pip install daytona` + `DAYTONA_API_KEY` | Cloud sandboxes; the backend Tmax's reported 27.2% used for eval. |
| `docker` | local | Fastest iteration; needs the Docker daemon + free disk. |

The scripts keep two timeouts ordered — **`RLLM_MODAL_SANDBOX_TIMEOUT_S`
(sandbox lifetime, 2400s) > `RLLM_HARNESS_RUN_TIMEOUT_S` (agent run cap,
1800s)** — so a long rollout still gets verified before its sandbox is reaped.
See terminal-rl's README for why making them equal causes
`NotFoundError: Sandbox has already shut down` storms.

## Files

| File | Description |
|------|-------------|
| `prepare_data.py` | Pull `tmax-15k` (train) + `terminal-bench@<ver>` (eval) |
| `train.py` | Load both datasets, select the harness, hand to `AgentTrainer` |
| `train_verl.sh` | **verl — full FT 9B, the faithful DPPO→GRPO reproduction** |
| `train_fireworks.sh` | Fireworks — managed LoRA-32 variant |
| `train_tinker.sh` | Tinker — single-machine LoRA-32 variant |
| `test.py` | Import/catalog/wiring smoke tests |
| `pyproject.toml` | Cookbook metadata |

## Citation

```bibtex
@misc{ivison2026tmaxsimplerecipeterminal,
      title={Tmax: A simple recipe for terminal agents},
      author={Hamish Ivison and Junjie Oscar Yin and Rulin Shao and Teng Xiao and Nathan Lambert and Hannaneh Hajishirzi},
      year={2026}, eprint={2606.23321}, archivePrefix={arXiv}, primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.23321}
}
```
