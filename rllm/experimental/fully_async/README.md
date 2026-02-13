# Fully Async Training

Fully asynchronous PPO training with decoupled rollout and training.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ RolloutExecutor │────▶│   MessageQueue   │────▶│ FullyAsyncTrainer│
│  (async rollout)│     │ (trajectory buf) │     │   (PPO update)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                                                 │
        │              ┌──────────────────┐               │
        └──────────────│   ParamSync      │◀──────────────┘
                       │ (weight updates) │
                       └──────────────────┘
```

**Key Components:**
- `rollout_executor.py` - Async rollout generation with SGLang backend
- `fully_async_trainer.py` - PPO trainer consuming from message queue
- `message_queue.py` - Trajectory buffer between rollout and training
- `param_sync.py` - Parameter synchronization from trainer to rollout
- `client.py` - HTTP client for rollout requests
- `metric_utils.py` - Metrics aggregation across training steps
- `utils.py` - Batch assembly and metric reduction utilities

## Installation

### 1. Create environment

```bash
micromamba create -n rllm-fully-async python=3.12 pip -c conda-forge
micromamba activate rllm-fully-async
pip install uv
```

### 2. Install dependencies

```bash
bash install_vllm_sglang_mcore_updated_sglang.sh
```

### 3. Install verl

```bash
git clone https://github.com/verl-project/verl.git
cd verl
git fetch
git checkout adff7956cefd8ef707cd67dd8e08c06fa63679bd
```

Apply the required patch:

```bash
git apply rllm/experimental/fully_async/verl_dp_actor.patch
```

Install:

```bash
uv pip install -e .
```

### 4. Install rllm

```bash
cd ~/rllm
uv pip install -e .
```

## Verl Patches

See `VERL_PATCHES.md` for details on required verl modifications.

**Patch file:** `verl_dp_actor.patch`

Changes to `verl/workers/actor/dp_actor.py`:
- Force single mini-batch for async training
- Token-mean loss scaling (instead of batch-size scaling)
- Guard against empty response_mask

## Running

TBD

## Configuration

See `config/` for example configurations.