# SWE Agent

Train `mini-swe-agent` on **swesmith** (synthetic Django/sympy/oauthlib bug-fixing tasks) and validate on **swebench-verified** (the human-labelled SWE-bench subset). Both datasets ship sandboxed: each task is a docker image with a broken repo + a `tests/test.sh` verifier, and the agent runs inside the container to produce a patch.

## Architecture

```
DatasetRegistry.load_dataset("swesmith", "default", as_tasks=True)          ┐
DatasetRegistry.load_dataset("swebench-verified", "default", as_tasks=True) ┘
                              │
                              ↓  rows wrapped as Tasks rooted at row's task_path
                              │
        AgentTrainer(agent_flow=mini-swe-agent)
                              │  (auto-detects SandboxedAgentFlow + harbor Tasks
                              │   → wires SandboxTaskHooks, pins gateway 127.0.0.1)
                              │
                              ├── AgentFlowEngine drives each rollout:
                              │     1. SandboxTaskHooks.setup → fresh sandbox, per-task verifier
                              │     2. mini-swe-agent.run() inside sandbox (LLM via gateway)
                              │     3. enrich_episode_with_traces from gateway
                              │     4. ctx.evaluator → tests/test.sh inside sandbox
                              │
                              └── Tinker (LoRA forward/backward + checkpoint)
```

## Quick start

```bash
uv pip install -e ".[tinker]"
rllm dataset pull harbor:swesmith
rllm dataset pull harbor:swebench-verified
bash cookbooks/swe_agent/train_tinker.sh
```

Equivalent CLI one-liner (no cookbook script needed since `mini-swe-agent` is built-in):

```bash
rllm train harbor:swesmith --agent mini-swe-agent \
    --val-dataset swebench-verified \
    --model Qwen/Qwen3-30B-A3B \
    --batch-size 2 --group-size 4
```

The CLI does everything the cookbook's `train.py` does. Reach for the cookbook when you need programmatic control over the loop, want to fork it for a custom training schedule, or want a Hydra config layered on top.

## What `train.py` looks like

```python
@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("swesmith", "default", as_tasks=True)
    val_dataset = DatasetRegistry.load_dataset("swebench-verified", "default", as_tasks=True)

    AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=load_agent("mini-swe-agent"),
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ).train()
```

That's the whole training script — same surface area as `cookbooks/math/train.py`. Three things make this work for sandbox flows without any extra plumbing in the user script:

1. **`as_tasks=True`** on `DatasetRegistry.load_dataset` wraps each harbor row as a Task rooted at its `task_path` and merges per-task `task.toml` metadata. Without it the engine can't find the per-task verifier or sandbox config.
2. **`AgentTrainer` auto-wires `SandboxTaskHooks`** when it sees a `SandboxedAgentFlow` agent or a dataset whose Tasks carry `task_path`. The hook gives each rollout its own sandbox + per-task `tests/test.sh` verifier (without it, parallel rollouts would clobber a shared `_sandbox` slot — same concurrency bug fix as eval got in #566, now ported into training).
3. **`AgentTrainer` pins `rllm.gateway.host=127.0.0.1`** when it auto-wires hooks. The harness rewrites loopback URLs to `host.docker.internal` so containers can call back to the gateway; the default routable-IP host wouldn't be rewritten and the agent's LLM calls would dead-end.

Override any of these by passing `hooks=`, `evaluator=`, or `rllm.gateway.host=...` explicitly.

## Modal mode (faster than local Docker)

Local Docker on a laptop is the bottleneck on this workload — every rollout pays ~10–15 min for image pull + agent install + verifier setup. Modal Sandboxes let those run in parallel in Modal's cloud; each rollout's wall-clock stays the same but you can run 32–64 in parallel without hammering one Mac.

```bash
brew install cloudflared        # macOS — Linux/Win: see docs link below
modal setup                     # first time only — opens browser for auth

bash cookbooks/swe_agent/train_modal.sh
```

Equivalent CLI:

```bash
rllm train harbor:swesmith --agent mini-swe-agent \
    --val-dataset swebench-verified \
    --model Qwen/Qwen3-30B-A3B \
    --batch-size 2 --group-size 4 \
    --sandbox-backend modal \
    --sandbox-concurrency 64
```

What `--sandbox-backend modal` triggers automatically:

1. **Sandboxes run on Modal**, not Docker. `SandboxTaskHooks` passes `sandbox_backend="modal"` into `_create_sandbox_for_task`, which falls through to `ModalSandbox(image=task.metadata["docker_image"])`. Modal's `Image.from_registry()` pulls the swebench image from Docker Hub (cached across containers, so cold-start is one-time per image).
2. **A cloudflared tunnel is spawned for the gateway.** The training gateway lives on `127.0.0.1:9090` on your laptop. Modal sandboxes can't reach that directly — they need a public URL. `AgentTrainer` detects `sandbox_backend=modal` (a non-local backend) and sets `rllm.gateway.tunnel="cloudflared"`. `GatewayManager.start()` runs `cloudflared tunnel --url http://127.0.0.1:9090`, scrapes the assigned `*.trycloudflare.com` URL, and stamps it onto `self.public_url`. The harness then ships that public URL into the sandbox via `OPENAI_API_BASE` so the in-container agent calls back to the gateway.
3. **Concurrency raised** via `--sandbox-concurrency 64`. The default `max_concurrent=4` on `MiniSweAgentHarness` is right for one local Docker; Modal scales out and you want it higher.

To use a fixed public URL (e.g. you're on a cloud GPU box already exposed at `https://my.host.com`), pass `rllm.gateway.public_url=https://my.host.com` and the tunnel auto-spawn skips.

> **Cost**: Modal bills per sandbox-second. Each rollout holds a sandbox for ~10–15 min. With `--group-size 4 --batch-size 2 = 8 rollouts/step`, that's roughly 1–2 sandbox-hours per training step at the per-rollout pricing. On crashes the trainer's `atexit` hook terminates live `ModalSandbox` instances; on `kill -9` you may need `modal app stop rllm-sandbox`.

> **Tunnel reachability**: cloudflared quick tunnels publish a URL immediately but DNS propagation takes 30–90 seconds. The first rollout may see a short DNS hiccup before settling.

[Cloudflared install docs](https://developers.cloudflare.com/cloudflare-tunnel/downloads/)

## Cost knobs

mini-swe-agent on a swebench task takes **~10–15 min wall-clock** (Docker image pull + agent install + LLM turns + pytest install + test run). With swebench-verified at 500 tasks and a default `test_freq=5`, every validation pass at default settings is ~75–125 hours. Use these to keep iteration short:

| Knob | Default | Recommended for smoke tests |
|---|---|---|
| `data.val_batch_size` | full dataset | `8` (one mini-batch) |
| `rllm.trainer.test_freq` | `5` | `20` or `0` (disable periodic val) |
| `--max-examples` | full | `1` for rollouts |
| `--max-steps` | epochs × len(loader) | `1` for "does it run end-to-end" |

For a fast end-to-end check (~20 min):

```bash
rllm train harbor:swesmith --agent mini-swe-agent --model Qwen/Qwen3-30B-A3B \
    --batch-size 1 --group-size 1 --max-steps 1 --max-examples 1
```

Always clean orphan sandboxes between runs:

```bash
docker rm -f $(docker ps -q --filter name=rllm-sandbox)
```

## Files

| File | Description |
|------|-------------|
| `train.py` | Hydra-driven Python training entry point |
| `train_tinker.sh` | Convenience wrapper with the recommended Tinker overrides (local Docker) |
| `train_modal.sh` | Same recipe, swap to Modal sandboxes + cloudflared tunnel |
| `README.md` | This file |

No `pyproject.toml` — `mini-swe-agent` is a built-in harness in `rllm.harnesses`, no plugin discovery needed.

## Verified

End-to-end run with `--batch-size 1 --group-size 1 --max-steps 1 --max-examples 1` on `harbor:swesmith` (Qwen3-30B-A3B): step 1 completes with 11–15 trace steps captured, `env_done=1.0`, `time/step≈800-880s`, `time/save_checkpoint≈2-4s`. Reward is 0.0 (untrained 30B model on an SWE bug — expected; the point is the pipeline works end-to-end).
