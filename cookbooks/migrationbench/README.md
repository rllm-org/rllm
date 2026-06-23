# MigrationBench Agent (AWS AgentCore Runtime)

Train a Java 8→17 migration agent on [MigrationBench](https://github.com/amazon-science/MigrationBench) with rLLM. The agent runs **inside an AWS Bedrock AgentCore Runtime** (a serverless remote runtime), not in-process — rLLM drives rollouts over the network and trains the policy locally with the **verl** backend.

## Overview

The agent code lives in the [agentcore-rl-toolkit](https://github.com/awslabs/agentcore-rl-toolkit) repo, not here. You build and deploy it from there ([`examples/strands_migration_agent`](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/examples/strands_migration_agent)), which produces the inputs this cookbook needs:

- **An AgentCore agent runtime ARN** — the deployed container where the agent loop to perform migrations is hosted.
- **A data S3 bucket** — holds the prepared MigrationBench repo tarballs and their `metadata.json`, written by the toolkit's `preprocess.py`. The container downloads repos from here at runtime, and `prepare_migrationbench_data.py` reads its metadata to register the dataset.
- **An output S3 bucket** — where the agent writes rollout results (rewards, etc.); rLLM polls it for each rollout. This can be a different bucket from the data bucket.

This cookbook only handles the rLLM side: registering the dataset from the data bucket's metadata and launching verl training against the remote runtime. The repo tarballs stay in S3 and are pulled at runtime by the container.

```
agentcore-rl-toolkit (separate repo)         rLLM (this cookbook)
──────────────────────────────────          ─────────────────────────
build + deploy agent  ──► agent ARN ─────►   train_…_verl.sh  (drive rollouts + train)
preprocess.py         ──► data bucket ───►   prepare_migrationbench_data.py  (register dataset)
                          output bucket ◄──  agent writes rewards; rLLM polls
```

> **Note:** this example supports **verl only** (distributed multi-GPU). It is tested on a single 8×B200 node with verl 0.8.0 / Qwen3-Coder-30B-A3B-Instruct (LoRA).

## Installation

```bash
# rLLM + verl backend (vLLM) + AgentCore runtime client
uv pip install -e ".[verl,agentcore]"

# Megatron deps for the verl training backend
bash scripts/install_megatron.sh <cu129|cu130|...>
```

## Prerequisites (agentcore-rl-toolkit)

Clone the toolkit, then preprocess the dataset and build/deploy the agent. Follow the
[strands_migration_agent README](https://github.com/awslabs/agentcore-rl-toolkit/tree/main/examples/strands_migration_agent):

```bash
git clone https://github.com/awslabs/agentcore-rl-toolkit
cd agentcore-rl-toolkit/examples/strands_migration_agent

# Upload prepared MigrationBench repos + metadata to your data bucket
python preprocess.py --s3-bucket-name <data-bucket>

# Build + deploy the agent container (see the example README) → records the agent runtime ARN
```

When this finishes you have the **agent runtime ARN**, the **data bucket**, and an **output bucket** — the only inputs the rest of this cookbook needs. Run the remaining steps from this cookbook folder (`cd cookbooks/migrationbench`); the train script sources `.env` from the current directory. Copy the example and fill it in:

```bash
cp .env.example .env
# edit .env:
#   AGENTCORE_AGENT_ARN=arn:aws:bedrock-agentcore:<region>:<account>:runtime/<name>
#   AGENTCORE_S3_BUCKET=<output-bucket>
```

## Dataset

Run from this cookbook folder (`cd cookbooks/migrationbench`):

```bash
python prepare_migrationbench_data.py --s3-bucket-name <data-bucket>
```

This downloads only the small `metadata.json` files from `s3://<data-bucket>/tars/{train,test}/`, then registers `migration_bench/{train,test}` with the rLLM `DatasetRegistry`:

- **Train** — repos under `tars/train/` with `num_test_cases > 0`.
- **Test** — all repos under `tars/test/`.

## Training

From this cookbook folder (`cd cookbooks/migrationbench`):

```bash
bash train_agentcore_migrationbench_verl.sh
```

The script sources `.env` for the agent ARN and output bucket, then runs verl with `rllm.remote_runtime.backend=agentcore` so rollouts execute in the deployed container. Tune `MODEL_PATH`, parallelism (`TP`/`EP`/`CP`), batch sizes, and `trainer.n_gpus_per_node`/`nnodes` in the script to match your hardware.

## Files

| File | Description |
|------|-------------|
| `prepare_migrationbench_data.py` | Register `migration_bench/{train,test}` from the data bucket's metadata |
| `train_agentcore_migrationbench_verl.py` | Python API training entrypoint (Hydra config, verl backend) |
| `train_agentcore_migrationbench_verl.sh` | Launch verl training against the AgentCore Runtime |
