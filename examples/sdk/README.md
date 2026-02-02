# SDK Trainer

This example demonstrates how to use the SDK Trainer for reinforcement learning with language models.

## Prerequisites

### 1. Install Verl

Run the installation script:
```bash
bash scripts/install_verl.sh
```

**Important:** Make sure to install `torch==2.6.0` when installing Verl. After `install_verl.sh` finishes, install `vllm==0.10.0`. You should see your torch version bumped to 2.7.1 after this - this is expected behavior.

**Troubleshooting:**
- If you encounter issues with `flash_attn`, reinstall it with:
  ```bash
  pip install flash-attn --no-build-isolation
  ```

- If you encounter errors with Ray, try:
  ```bash
  pip install ray==2.48.0
  ```

## Running the Examples

### Hendrycks Math Training

This is the simplest example with a single agent and single turn.

```bash
./train_hendrycks_math.sh
```

### Solver-Judge Flow Training

This is a more complex example with 2 agents and more complex grouping logic.

```bash
./train_solver_judge_flow.sh
```

## Optional: Manual Proxy Setup

By default, the training scripts use `rllm.sdk.proxy.mode=subprocess` which automatically manages the LiteLLM proxy. If you prefer to manually manage the proxy, you can set `rllm.sdk.proxy.mode=external` in your training script and start the proxy yourself:

```bash
#!/bin/bash

# Set ulimit first
ulimit -n 65536

# Set aiohttp connection limits
export AIOHTTP_CONNECTOR_LIMIT=4096
export AIOHTTP_KEEPALIVE_TIMEOUT=60

# Verify the limits are set
echo "Current ulimit -n: $(ulimit -n)"
echo "AIOHTTP_CONNECTOR_LIMIT: $AIOHTTP_CONNECTOR_LIMIT"
echo "AIOHTTP_KEEPALIVE_TIMEOUT: $AIOHTTP_KEEPALIVE_TIMEOUT"
echo "Starting LiteLLM proxy..."

# Start the proxy
python -m rllm.sdk.proxy.litellm_server \
  --config litellm_proxy_config_autogen.yaml \
  --host 127.0.0.1 \
  --port 4000 \
  --state-dir /tmp/litellm_proxy \
  --cs-endpoint http://localhost:8000 \
  --cs-api-key "your-api-key-here" \
  --project rllm-agent-sdk-engine \
  --admin-token my-shared-secret
```