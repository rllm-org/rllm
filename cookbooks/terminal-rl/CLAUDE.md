# terminal-RL cookbook — working notes

Agentic RL on terminal-bench tasks. The local machine is a **coordinator**, not
a trainer: weights, optimizer state and checkpoints live on Fireworks, rollouts
run in Modal sandboxes. No GPU is needed locally.

Flow: `train_fireworks_debug.sh` → `train_debug.py` → Fireworks RLOR trainer job
\+ a rollout deployment; agents in Modal sandboxes reach the local model gateway
through a public URL (tunnel or routable IP).

See `deploy/` for unattended operation on a server.

## Launching

The committed script is a starting point, not a complete command. Two overrides
are effectively required:

```bash
bash train_fireworks_debug.sh \
  rllm.gateway.tunnel=<live-url> \        # baked-in URL is a placeholder
  rllm.trainer.experiment_name=<name>     # see naming below
```

Hydra is last-wins, and the script forwards `"$@"`, so appended flags override
the file. Prefer overriding to editing the file.

`FIREWORKS_API_KEY` must match the account you intend. The script falls back to
`~/.rllm/config.json` → `api_keys.fireworks`, and the shell profile may export a
key for a *different* account — an easy way to train on the wrong one silently.

## Experiment naming

Promoted checkpoints are `<experiment_name>-step-<n>`, so two runs sharing a
name overwrite each other's models with no error. Always encode the config:

```
<base>-r<rank>-lr<lr>[-a<alpha>][-<host-tag>]
qwen3p5-35b-a3b-tb-v2-debug-r128-lr1p5e-4
```

Names must be id-safe: lowercase, hyphens, **no dots** (`1.5e-4` → `1p5e-4`).
Budget ~63 chars including the `-step-<n>` suffix. Point
`episode_log_dir` / `backend_batch_log_dir` at the same name so logs, wandb runs
and models stay aligned.

## Failure modes that look like success

These cost hours each; none of them announce themselves.

- **Orphaned gateway workers.** `rllm_model_gateway` processes outlive their
  parent `train_debug.py`. If they keep the gateway port, the next run looks
  perfectly healthy — trainer `RUNNING`, episodes completing — but every episode
  returns `No traces found` and is filtered before training. Symptom:
  `consumed=2, filtered=21` and zero optimizer steps. Always
  `pkill -f rllm_model_gateway` after killing a run.
- **Trainer jobs keep billing.** Killing the local process does not stop the
  remote job; it lingers until its own inactivity timeout (up to 2h). Delete it
  explicitly, **by exact id** — the account is shared, so never "delete all
  active jobs".
- **Pruning empty log dirs kills the run.** The episode logger uses
  `open(path, "w")`, which does not create parents. Remove log *files* only;
  deleting `episodes/` or `backend_batches/` between steps raises
  `FileNotFoundError` and takes the run down.
- **Episode logging fills disks.** ~12 GB/h at `n_parallel_tasks=192`; it filled
  a 460 GB volume overnight. `log_backend_batches` is the bulk (~100–200 MB per
  compressed batch file).
- **Placement can hang for an hour.** A trainer job stuck in `CREATING` fails
  the client's 3600s timeout having produced nothing. This is *account
  capacity*, not config: rank 128 hung repeatedly on one account and placed in
  ~14 min on another.

## Numbers worth knowing

- `tb_v2_debug` is **8 prompts**. With `group_size=16` and `train_batch_size=1`,
  one optimizer step ≈ 128 episodes and the set cycles every 8 steps, so
  per-step reward is jumpy by construction. `Tasks: n/8000` is 8 × 1000 epochs.
- Throughput: ~14 min/step at rank 32, ~27 min/step at rank 128.
- Healthy placement reports `healthz=OK` in 12–15 min.
- Baseline solve rate on these 8 prompts is ~45–50% with reward std ~0.4, which
  is near-ideal for GRPO (`filter_uniform_groups` discards little).

## LoRA alpha

`lora_alpha` defaults to **32 regardless of rank**, so scaling `alpha/r` is 1.0
at rank 32 but 0.25 at rank 128 — a 4× smaller effective update at the same
learning rate. Set it explicitly with `+model.lora_alpha=<int>` (Hydra append;
the key is not in rllm's schema). The plumbing lives in
`rllm/trainer/fireworks/fireworks_backend.py`; without it the value is silently
ignored.

## Resuming

`save_freq=10` checkpoints every 10 optimizer steps. To continue rather than
restart:

```
training.resume_from_fireworks_job_id=training-api-service-xxxxxxxx
training.resume_from_dcp_checkpoint=<snapshot-name>
```

Only works into the **same LoRA rank** — a rank-32 checkpoint cannot seed a
rank-128 run.
