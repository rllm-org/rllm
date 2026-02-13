# Remote Agent Training

Run your agent code on your **local desktop** while rLLM's GPU training infrastructure runs on a **remote cluster** (Modal, RunPod, bare-metal, etc.). No need to bundle your agent environment, tools, or reward logic into the GPU container.

## Architecture

```
Local Desktop                          Remote GPU Cluster
┌───────────────────────┐              ┌──────────────────────────┐
│  AgentTrainerClient   │              │  TrainingServer (FastAPI) │
│  ├─ rollout_fn        │  HTTP        │  ├─ SGLang Servers       │
│  ├─ environments      │◄────────────►│  ├─ FullyAsyncTrainer    │
│  ├─ reward functions  │              │  ├─ ParameterSynchronizer│
│  └─ dataset           │              │  └─ MessageQueue         │
└───────────────────────┘              └──────────────────────────┘
```

## Prerequisites

**GPU server** (inside the rLLM Docker container or environment):
- rLLM installed with all GPU dependencies (verl, SGLang, PyTorch, etc.)
- Ray cluster running
- A training config YAML file

**Local desktop** (lightweight):
- `pip install rllm` (or editable install of this repo)
- `pip install httpx transformers torch` (no GPU needed)
- Network access to the GPU server

## Quick Start

### Step 1: Start the Training Server (GPU machine)

SSH into your GPU machine or deploy via your cloud platform. The server uses Hydra for configuration, inheriting from the built-in `ppo_trainer` defaults:

```bash
# On the GPU machine (inside rLLM environment)
python examples/fully_async/remote_training/server.py \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    +server.host=0.0.0.0 \
    +server.port=8000
```

You can pass any Hydra override on the command line (same syntax as the colocated training scripts):

```bash
python examples/fully_async/remote_training/server.py \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    trainer.n_gpus_per_node=8 \
    rollout.n_gpus_per_node=4 \
    +server.host=0.0.0.0 \
    +server.port=8000
```

The server starts a FastAPI application and waits for a client to connect. It does **not** initialise GPU components until a client calls `/v1/configure`, so you can start it before deciding on experiment hyperparameters.

### Step 2: Run the Agent Client (local desktop)

```bash
# On your local desktop
python client.py \
    --server-url http://<GPU_SERVER_IP>:8000 \
    --model-name Qwen/Qwen2.5-7B \
    --dataset-name my_dataset \
    --dataset-split train \
    --n 4 \
    --max-concurrency 128
```

Replace `<GPU_SERVER_IP>` with the actual IP address or hostname of the GPU machine. If using Modal or another platform with a public URL, use that URL instead (e.g. `https://my-app--serve.modal.run`).

### What happens

1. **Client connects** to the server and sends config overrides (learning rate, project name, etc.)
2. **Server initialises** SGLang inference servers, loads the model, sets up the trainer
3. **Server starts training** -- the trainer loop begins consuming from the message queue
4. **Client runs rollouts** locally -- calls `rollout_fn` concurrently, which uses `client.chat_completion()` to generate responses via the remote server
5. **Client submits trajectories** to the server as JSON over HTTP
6. **Server trains** on received trajectories, periodically syncing weights to SGLang
7. **Client waits** for the server to signal training completion

## Writing Your Own Agent

The `client.py` example shows the minimal structure. The key is the `rollout_fn`:

```python
async def rollout_fn(client, tokenizer, **kwargs):
    """
    Args:
        client:    RemoteRolloutClient (same interface as RolloutClient)
        tokenizer: HuggingFace tokenizer (loaded locally)
        **kwargs:  One row from your dataset
    Returns:
        Trajectory with reward and metadata
    """
    # Generate a response using the remote model
    messages = [{"role": "user", "content": kwargs["question"]}]
    response_msg, output = await client.chat_completion(messages)

    # Compute reward locally (your custom logic)
    reward = my_reward_function(response_msg, kwargs["ground_truth"])

    # Build and return trajectory
    trajectory = Trajectory(sequences=[output.to_sequence()], reward=reward)
    trajectory.metadata = {"reward": reward}
    return trajectory
```

This is **the same interface** as the colocated `AsyncAgentTrainer`. You can reuse existing `rollout_fn` implementations -- just swap `AsyncAgentTrainer` for `AgentTrainerClient`.

## Config Overrides

The client can override training hyperparameters without modifying the server's config file:

```python
trainer = AgentTrainerClient(
    server_url="http://gpu-server:8000",
    rollout_fn=rollout_fn,
    model_name="Qwen/Qwen2.5-7B",
    dataset=dataset,
    config_overrides={
        # Algorithm
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,

        # Training
        "async_training.required_samples": 64,
        "async_training.trigger_parameter_sync_step": 2,

        # Logging
        "trainer.project_name": "my-project",
        "trainer.experiment_name": "run-42",
        "trainer.save_freq": 10,

        # Rollout
        "rollout.total_rollout_steps": 5000,
        "rollout.n": 4,
    },
)
```

Hardware-bound keys (`trainer.n_gpus_per_node`, `rollout.nnodes`, `actor_rollout_ref.model.path`) are protected and cannot be overridden from the client.

## API Reference

The TrainingServer exposes these HTTP endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/configure` | Send config overrides, initialise GPU components |
| POST | `/v1/start` | Start the training loop |
| POST | `/v1/generate` | Proxy generation to SGLang (returns 503 during weight sync) |
| POST | `/v1/trajectories` | Submit a `TrajectoryGroup` as JSON |
| GET | `/v1/status` | Training status: `param_version`, `is_syncing`, `queue_size`, `training_complete` |
| GET | `/v1/config` | Public config info: model path, context lengths, `n`, `required_samples` |

## End-to-End Example: Solver-Judge on Countdown

The `solver_judge/` subdirectory contains a complete, runnable example adapted from `examples/solver_judge/`. It trains a model on the [Countdown](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) math task using a Solver-Judge workflow where:

- **Solver** generates multiple candidate solutions in parallel
- **Judge** reviews all candidates and selects the best one
- **Reward** is computed locally (did the judge pick a correct solution?)

### Step 0: Prepare the dataset (once, on your local desktop)

```bash
python -m examples.fully_async.remote_training.solver_judge.prepare_data \
    --output-dir ./countdown_data
```

This downloads from HuggingFace and saves `train.json` / `test.json` locally.

### Step 1: Start the server (GPU machine)

```bash
# On the GPU machine (8x A100 / H100)
python examples/fully_async/remote_training/server.py \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    +server.host=0.0.0.0 \
    +server.port=8000
```

### Step 2: Run the Solver-Judge client (local desktop)

```bash
python -m examples.fully_async.remote_training.solver_judge.train \
    --server-url http://<GPU_SERVER_IP>:8000 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --data-dir ./countdown_data \
    --n-solutions 2 \
    --n 4 \
    --max-concurrency 128
```

This runs the full Solver-Judge workflow locally. For each problem:
1. The **Solver** calls `client.chat_completion()` twice (generating 2 candidates)
2. The **Judge** calls `client.chat_completion()` once (selecting the best)
3. The reward is computed locally using the Countdown scoring function
4. The trajectory (all 3 sequences + reward) is submitted to the server

The server trains on the trajectories with GRPO, periodically syncing weights to SGLang. The client's generation requests automatically use the updated model.

---

## Deployment Examples

### Bare-metal / SSH

```bash
# Terminal 1 (GPU machine)
ssh gpu-machine
cd /path/to/rllm
python examples/fully_async/remote_training/server.py \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    +server.port=8000

# Terminal 2 (local desktop)
python examples/fully_async/remote_training/client.py \
    --server-url http://gpu-machine:8000 \
    --model-name Qwen/Qwen2.5-7B \
    --dataset-name my_dataset
```

### With SSH port forwarding

If the GPU machine is behind a firewall:

```bash
# Forward local port 8000 to gpu-machine:8000
ssh -L 8000:localhost:8000 gpu-machine \
    "cd /path/to/rllm && python examples/fully_async/remote_training/server.py \
     actor_rollout_ref.model.path=Qwen/Qwen2.5-7B +server.port=8000"

# Then connect to localhost
python client.py --server-url http://localhost:8000 ...
```

### Modal

```python
# modal_app.py
import modal

app = modal.App("rllm-training")
image = modal.Image.from_dockerfile("Dockerfile")

@app.function(gpu="A100:8", image=image, timeout=86400)
@modal.web_server(port=8000)
def serve():
    from omegaconf import OmegaConf
    from rllm.experimental.fully_async.remote import TrainingServer
    # For Modal, pass a pre-built config directly
    config = OmegaConf.load("/config.yaml")
    server = TrainingServer(config=config)
    server.run(host="0.0.0.0", port=8000)
```

```bash
# Deploy
modal deploy modal_app.py

# Run client against the Modal URL
python client.py \
    --server-url https://your-org--rllm-training-serve.modal.run \
    --model-name Qwen/Qwen2.5-7B \
    --dataset-name my_dataset
```
