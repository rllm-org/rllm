# terminal-RL coordinator deployment

Run the terminal-RL training loop unattended on a CPU-only box.

This host is a **coordinator**, not a trainer. Weights, optimizer state and
checkpoints live on Fireworks; rollouts run in Modal sandboxes. Locally you
only run the model gateway and the workflow engine, so no GPU is required.

Recommended: 4–8 vCPU, 16 GB RAM, 100 GB+ disk.

## Contents

| file | purpose |
|---|---|
| `setup.sh` | clone + editable-install the five local packages, verify imports |
| `sweep.sh` | sequential hyperparameter sweep, one job at a time |
| `supervisor.sh` | keep a single config training indefinitely |
| `terminal-rl-sweep.service` | systemd user unit (reboot persistence) |
| `terminal-rl.env.example` | credentials/config template → `~/.rllm/terminal-rl-auto.env` |

## Why a server beats a laptop

Every failure that cost real time was environmental, not algorithmic:

- reboots and sleep killed detached runs
- a router-level DNS filter blocked `*.trycloudflare.com`, so quick tunnels
  could not even be provisioned
- quick-tunnel hostnames expire, silently stranding remote sandboxes

On a box with a routable address, set `TERMINAL_RL_GATEWAY_MODE=public` and
skip tunnels entirely — this is what `rllm.gateway.tunnel=http://<ip>:<port>`
in the committed script was always for.

## Migration checklist

1. **Carry uncommitted work.** Branch `tianyi/terminal-rl` has it; otherwise
   `git diff > migrate.patch` for `cookbooks/terminal-rl/train_fireworks_debug.sh`
   and `rllm/trainer/fireworks/fireworks_backend.py`.
2. **Check out the Fireworks monorepo** at `~/fireworks`. Three editable
   installs come from it; without them `import training.provision` fails.
3. `bash setup.sh` — installs and *verifies* each import resolves to a
   checkout rather than a shadowing wheel.
4. **Copy credentials**: `~/.modal.toml` (with the intended profile active),
   the Fireworks key into `~/.rllm/terminal-rl-auto.env`, wandb via `~/.netrc`.
5. **Copy the dataset**: `~/.rllm/datasets/tb_v2_debug` (56 KB), or regenerate
   with `prepare_data.py`.
6. **Open the gateway port** in the firewall/security group if using
   `public` mode.
7. **Stop the old host first.** Two coordinators means two billing trainer
   jobs and two gateways competing for the same rollouts.
8. `systemctl --user enable --now terminal-rl-sweep`, then
   `loginctl enable-linger $USER`.

### Resuming instead of restarting

With `save_freq=10` a run checkpoints every 10 optimizer steps, so a migration
mid-config does not have to throw away progress:

```
training.resume_from_fireworks_job_id=training-api-service-xxxxxxxx
training.resume_from_dcp_checkpoint=<snapshot-name>
```

Only works from step 10 onward, and only into the **same LoRA rank** — a
rank-32 checkpoint cannot seed a rank-128 run.

## Hard-won behaviour encoded in these scripts

Each of these cost hours; the guards exist for a reason.

- **Gateway workers outlive their parent.** Killing `train_debug.py` leaves
  `rllm_model_gateway` processes holding the port. The next run then looks
  perfectly healthy — trainer `RUNNING`, episodes completing — while every
  episode returns `No traces found` and is filtered before training. One run
  burned 1h50m and 383 episodes this way. Both scripts kill orphans before
  launching.
- **Never delete empty log directories.** The episode logger opens files with
  `open(path, "w")`, which does not create parents; if a prune removes
  `episodes/` or `backend_batches/` between steps, the run dies with
  `FileNotFoundError`. Prune **files only**.
- **Trainer jobs survive the local process.** Stopping a run does not stop
  billing; jobs linger until their own inactivity timeout. Both scripts delete
  the job they created, **by exact id** — never "all active jobs", since the
  account is shared and that would kill a colleague's run.
- **`pgrep` can fail outright** (macOS sysmond entitlement loss). A failed
  liveness check must not be read as "training is down", or you get duplicate
  launches. Both scripts fall back to `ps`.
- **Episode logging is the disk hog**, not model state: ~12 GB/h at
  `n_parallel_tasks=192`, and it filled a 460 GB volume overnight. Pruning
  holds it near 5 GB.
- **Placement can hang for an hour.** A trainer job stuck in `CREATING`
  eventually fails the client's 3600s timeout having produced nothing, so the
  sweep abandons an attempt after `PLACE_LIMIT_MIN`.

## Tuning notes

- `lora_alpha` defaults to **32** regardless of rank (`DEFAULT_LORA_ALPHA`), so
  the LoRA scaling `alpha/r` is 1.0 at rank 32 but only **0.25** at rank 128 —
  a 4× smaller effective update at the same learning rate. Set it explicitly
  with `TERMINAL_RL_EXTRA_ARGS='+model.lora_alpha=256'` (the plumbing for this
  lives in `fireworks_backend.py`; without it the value is silently ignored).
- `tb_v2_debug` is **8 prompts**. With `group_size=16` and
  `train_batch_size=1`, one optimizer step is 16 rollouts of a single task and
  the set cycles every 8 steps, so per-step reward is jumpy by construction.
- Throughput observed: ~128 episodes per optimizer step; ~14 min/step at rank
  32, ~27 min/step at rank 128. Budget 150 steps ≈ 68h at rank 128.
