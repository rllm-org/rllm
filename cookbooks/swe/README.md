# SWE Agent

Software-engineering agent flow for rLLM that runs **mini-swe-agent** inside
Modal sandboxes, generates patches, and grades them against SWE-style
benchmarks.

## Overview

This cookbook keeps the original SWE package structure while fitting the rLLM
cookbook layout. The agent uses native OpenAI-compatible tool calling for bash
commands, supports context compaction, records clean trajectories, and can run
unchanged for evaluation or veRL training through the rLLM gateway.

Supported datasets:

| Dataset | Eval type | Notes |
|---|---|---|
| `swe_bench_pro` | `swebench_pro` | Uses the SWE-bench Pro harness from the cookbook submodule |
| `swe_bench_multilingual` | `swebench` | Uses the installed `swebench` harness |
| `swe_smith*` | `swesmith` | Uses the installed `swesmith` package, removes tests during rollout, and grades in a fresh sandbox |
| `swe_rebench_v2*` | `swe_rebench_v2` | Reuses the rollout sandbox for grading; set `SWE_REBENCH_V2_PATH` to a local checkout for parser code |

## Architecture

```
SWEAgentFlow.run(task, config)
  |
  +-- create Modal sandbox from task["docker_image"]
  +-- run mini-swe-agent DefaultAgent with OpenAIClientModel
  |     |
  |     +-- chat.completions.create(..., tools=[bash])
  |     +-- execute bash action in sandbox
  |     +-- compact context when configured token limits are reached
  |
  +-- return Episode with patch, exit_status, messages, segments

SWEEvaluator.evaluate(task, episode)
  |
  +-- route by task["eval_type"]
  +-- run dataset-specific grader
  +-- return EvalOutput(reward, is_correct, pass/fail signals)
```

## Installation

From the rLLM repo root:

```bash
uv pip install -e ".[verl,swe]"
uv pip install -e cookbooks/swe
git submodule update --init --recursive cookbooks/swe/external/SWE-bench_Pro-os
```

SWE-bench Pro is intentionally the only submodule in this cookbook. SWE-smith
is installed as a package dependency. SWE-rebench V2 is optional; point
`SWE_REBENCH_V2_PATH` at a local checkout before evaluating those tasks.

Modal authentication is required for live rollouts and grading:

```bash
modal setup
```

Create a `.env` file in `cookbooks/swe/` or the process environment with the
API keys used by the model endpoint and Modal.

## Qwen3.5 veRL Environment

The 9B Megatron training launchers expect a veRL/vLLM environment with Qwen3.5
support. The default cookbook-local environment path is:

```bash
cookbooks/swe/.venv-verl-vllm018/.venv
```

Build or refresh it with the cookbook setup script:

```bash
bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
```

The script installs the pinned Qwen3.5 stack used by these launchers:
`torch==2.10.0+cu129`, `vllm==0.18.0`, `transformers==5.3.0`,
`megatron-core==0.17.0`, `megatron-bridge==0.4.0`,
`flash-attn==2.8.3`, TransformerEngine `release_v2.12`, Apex,
`flash-linear-attention==0.4.1`, NVIDIA ModelOpt, veRL, rLLM, the model
gateway, and this cookbook.

Useful overrides:

```bash
VENV_ROOT=/path/to/venv-root bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
VERL_PATH=/path/to/verl bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
MAX_JOBS=64 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
FLASH_ATTN_CUDA_ARCHS=100 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
NVTE_CUDA_ARCHS=100 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
RUN_SMOKE_TEST=0 bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
```

The source builds require a CUDA toolkit with `nvcc` on `PATH`; on the current
H100/B200 nodes `/usr/local/cuda/bin` is the expected source. The script unsets
`PYTHONPATH` while building so host packages do not leak into the venv.
For B200-only builds, keep `FLASH_ATTN_CUDA_ARCHS=100` and
`NVTE_CUDA_ARCHS=100`; the package defaults also compile older architectures
and are much slower.

On some B200 images, `ldconfig` resolves `libcuda.so.1` from the CUDA compat
directory before the real driver library. That makes Torch report CUDA error
803 even though `nvidia-smi` works. The setup and training scripts now prefer
`/usr/lib/x86_64-linux-gnu/libcuda.so.1` when it exists.

## Data

Prepare evaluation JSON:

```bash
python -m swe.prepare_data \
    --dataset swe_bench_pro \
    --split test \
    --output data/swe_bench_pro.json
```

Register datasets for rLLM training:

```bash
python -m swe.prepare_rllm_data --dataset swe_smith --split train
python -m swe.prepare_rllm_data --dataset swe_bench_multilingual --split test
python -m swe.scripts.prepare_filtered_mix
```

`prepare_rllm_data.py` writes to rLLM's `DatasetRegistry`.

## Eval

```bash
python -m swe.scripts.run_eval \
    --dataset swe_bench_pro \
    --model anthropic/claude-sonnet-4-5-20250929 \
    --slice 0:5 \
    --output_dir results/smoke \
    --verbose
```

For OpenAI-compatible hosted vLLM:

```bash
export HOSTED_VLLM_API_BASE=http://localhost:8000/v1
export HOSTED_VLLM_API_KEY=fake-api-key

python -m swe.scripts.run_n_eval \
    --dataset swe_smith_py \
    --model hosted_vllm/Qwen/Qwen3.5-9B \
    --n_runs 3 \
    --n_parallel 20 \
    --output_dir results/swe_smith_py_pass_at_3
```

## Training

The maintained launchers are the 9B Megatron runs:

```bash
bash cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh
bash cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron_h100.sh
```

Both scripts run `python -m swe.scripts.train_swe_verl` and accept Hydra
overrides as trailing arguments. They resolve paths relative to this cookbook
and expect a veRL-capable virtualenv to be active, available at `/tmp/verl_venv`,
available through `VENV_DIR`, available at the cookbook-local setup path, or
available under `verl/.venv` in the rLLM checkout.

The Qwen3.5-9B B200 script defaults to the full two-node Megatron shape:

```bash
NNODES=2
NGPUS_PER_NODE=8
ACTOR_TP=2
ACTOR_CP=2
ACTOR_PP=1
ROLLOUT_TP=1
```

It also expects a local Qwen3.5-9B snapshot, the filtered SWE-smith training
dataset in `DatasetRegistry`, Modal credentials for SWE-ReX sandboxes, and W&B
credentials when `LOGGER` includes `wandb`.

This runbook covers two launch styles:

- a normal Ray cluster where you already have one CPU driver and the B200 GPU
  workers allocated and reachable
- ByteDance Arnold launch, preferably using Arnold only for B200 workers while
  a pre-probed CPU driver owns Ray head, Modal, W&B, and checkpoints

It intentionally starts after worker allocation. Use your normal MLX/Arnold
reservation flow to get the CPU and B200 worker ids, then run the commands below
on those workers. The exact allocation command is infra-specific; the important
training requirement is the topology: one Modal-good CPU driver plus two
8-GPU B200 workers for the default Qwen3.5-9B run.

### Training Prerequisites

Commands in this section are from the rLLM repo root unless a ModelChef root is
explicitly called out. In a ModelChef checkout, the rLLM root is
`submodules/rllm`.

Preflight checklist:

- CPU driver has working Modal and W&B credentials.
- CPU driver passes the Modal/SWE-ReX dynamic probe below.
- Every Ray node has the veRL/vLLM runtime, rLLM checkout, SWE cookbook, model
  snapshot, and `RLLM_HOME` dataset registry.
- `ray status` on the CPU head shows the expected 16 B200 GPUs before training.
- Large outputs go to `/tmp`, HDFS, or another mounted volume, not the repo
  checkout.

Prepare the runtime on every CPU/GPU node that will run Ray processes:

```bash
cd /path/to/rllm
bash cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
```

On ByteDance machines with prebuilt SWE runtime artifacts, restore those
artifacts instead of rebuilding the venv and CUDA wheels:

```bash
cd /path/to/rllm
export RLLM_SWE_ARTIFACT_DIR=/path/to/rllm_swe_artifacts
source cookbooks/swe/launchers/swe_artifact_utils.sh
restore_swe_artifacts
```

The artifact bundle is expected to provide:

- `/tmp/verl_venv`
- Qwen3.5-9B cache under `/tmp/hf_cache`, or a mounted `MODEL_PATH`
- `RLLM_HOME` data, including the SWE-smith filtered mix
- the Megatron CP2 overlay required for `ACTOR_CP=2`

For non-artifact runs, set the model path explicitly on every node:

```bash
export MODEL_PATH=/path/to/Qwen3.5-9B/snapshot
```

Set credentials on the CPU driver. The training script reads Modal from the
environment or from `~/.modal.toml`; W&B is read from the environment or the
normal W&B netrc login.

```bash
modal setup
wandb login

# Equivalent non-interactive form:
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
export WANDB_API_KEY=...
```

Prepare the filtered SWE-smith mix if it is not already in restored
`RLLM_HOME`:

```bash
cd /path/to/rllm
python -m swe.scripts.prepare_filtered_mix
```

Before spending GPU time, probe the candidate CPU driver. This is mandatory for
SWE training because the CPU is selected for outbound Modal/SWE-ReX behavior,
not raw CPU speed. Static host/network shape is only a weak hint; CPUs from the
same collection can behave differently under Modal concurrency. Use only CPUs
whose probe summary has `ok == total == 12`, no failures, no transient
failures, and `create_env_s.p95 <= 30`.

```bash
cd /path/to/rllm
PYTHONPATH=$PWD:$PWD/cookbooks/swe \
SWE_REX_REMOTE_RETRIES=0 \
python -m swe.scripts.modal_swerex_reliability_test \
  --total 12 \
  --concurrency 6 \
  --mode light \
  --out /tmp/modal_probe_${USER}.jsonl
```

If the probe fails, discard that CPU driver and probe a different CPU before
starting B200 workers. In the all-Arnold CPU-head path, the entrypoint runs this
same gate and exits with `BAD_CPU_DRIVER` before training starts. A small probe
passing is necessary, but full training can still expose Modal transport errors
at higher load; if zero Modal transport errors are required, run a larger probe
closer to the intended rollout concurrency before committing GPU time.

### Normal Ray Cluster

Use this path when you already have a CPU driver plus GPU workers and do not
want Arnold to create the cluster. The CPU driver owns Modal, W&B, checkpoint
paths, and the training process. GPU workers only join Ray and run vLLM and
Megatron actors.

On the CPU driver:

```bash
cd /path/to/rllm
export RLLM_SWE_ARTIFACT_DIR=/path/to/rllm_swe_artifacts  # optional
source cookbooks/swe/launchers/swe_artifact_utils.sh      # optional
restore_swe_artifacts                                    # optional

export CPU_HEAD_IP=REPLACE_WITH_CPU_IPV4_OR_IPV6
ray stop --force || true
ray start \
  --head \
  --node-ip-address="$CPU_HEAD_IP" \
  --port=6379 \
  --ray-client-server-port=10001 \
  --dashboard-host=0.0.0.0 \
  --num-cpus=32 \
  --num-gpus=0 \
  --disable-usage-stats
```

On each 8-GPU worker:

```bash
cd /path/to/rllm
export RLLM_SWE_ARTIFACT_DIR=/path/to/rllm_swe_artifacts  # optional
source cookbooks/swe/launchers/swe_artifact_utils.sh      # optional
restore_swe_artifacts                                    # optional

export CPU_HEAD_IP=REPLACE_WITH_CPU_IPV4_OR_IPV6
ray stop --force || true

case "$CPU_HEAD_IP" in
  *:*) RAY_HEAD_ADDRESS="[$CPU_HEAD_IP]:6379" ;;
  *) RAY_HEAD_ADDRESS="$CPU_HEAD_IP:6379" ;;
esac

ray start \
  --address="$RAY_HEAD_ADDRESS" \
  --num-cpus=240 \
  --num-gpus=8 \
  --disable-usage-stats
```

Back on the CPU driver, verify that Ray sees the full GPU pool:

```bash
python - <<'PY'
import ray

ray.init(address="auto")
print(ray.cluster_resources())
ray.shutdown()
PY
```

Then launch training from the CPU driver:

```bash
cd /path/to/rllm
export RAY_ADDRESS=auto
export RLLM_RUN_TASK_RUNNER_LOCAL=1
export RLLM_RAY_WORKING_DIR=$PWD
export NNODES=2
export NGPUS_PER_NODE=8
export ACTOR_TP=2
export ACTOR_CP=2
export ACTOR_PP=1
export ROLLOUT_TP=1
export ROLLOUT_SKIP_MM_PROFILING=true
export ROLLOUT_DISTRIBUTED_EXECUTOR_BACKEND=uni
export MODEL_USE_REMOVE_PADDING=true
export RLLM_RAY_NOSET_CUDA_VISIBLE_DEVICES=1
export LOGGER='[console,wandb]'
export WANDB_MODE=online
export RLLM_SWE_REQUIRE_MEGATRON_CP2=1

# Keep large outputs off the repo filesystem.
export RLLM_SWE_OUTPUT_DIR=/tmp/rllm_swe_outputs
export CHECKPOINT_ROOT=$RLLM_SWE_OUTPUT_DIR/checkpoints
export TRAJ_DIR="$RLLM_SWE_OUTPUT_DIR/trajectories/\${trainer.experiment_name}"

bash cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh \
  ++trainer.total_training_steps=100
```

For one-step smoke tests, add smaller overrides:

```bash
bash cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh \
  train_max_samples=16 \
  val_max_samples=4 \
  actor_rollout_ref.rollout.n=1 \
  rllm.workflow.n_parallel_tasks=4 \
  ++trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=1000 \
  trainer.test_freq=1000 \
  trainer.val_before_train=false \
  ++rllm.trainer.val_before_train=false
```

### ByteDance Arnold Launch

Use Arnold launch for ByteDance-managed CPU and B200 allocation. In this mode
Arnold starts the CPU Ray head, starts the B200 Ray workers, runs the Modal CPU
probe on the selected CPU, restores the SWE runtime artifacts, and then launches
the same `run_swe_training_9b_megatron.sh` training script.

Experiment-specific changes should normally live in the Arnold YAML, not in the
training script. Use Arnold `envs` for launcher-level knobs and
`RLLM_SWE_FULL_EXTRA_ARGS` for Hydra overrides.

Available configs under `cookbooks/swe/launchers/`:

| Config | Use | Notes |
|---|---|---|
| `launch_swe_qwen35_9b_megatron_b200.yaml` | Main all-Arnold path | Arnold selects the CPU head and B200 workers. Use this as the base for full training. |
| `launch_swe_qwen35_9b_megatron_b200_external_ray_workers.yaml` | Diagnostic path | Arnold launches only B200 workers; a manually managed CPU Ray head is supplied externally. |
| `launch_swe_qwen35_9b_megatron_b200_external_driver.yaml` | Cluster-hold path | Arnold creates a CPU+B200 cluster and holds it for an external driver to connect by Ray Client. |

#### All-Arnold Full Training

The target production shape is one Arnold CPU head plus two 8-GPU B200 workers
on `cloudnative-useast1b`:

```text
CPU head: public_CPU
B200 workers: 2 x 8 GPUs
Cluster: cloudnative-useast1b
B200 queue: compute-598-useast1b-cloudnative-aioci-mlsys.inference-guarantee
Topology: ARNOLD_NETWORK_TOPOLOGY=SAME_NETMINIPOD
Trainer shape: NNODES=2, NGPUS_PER_NODE=8
Megatron shape: ACTOR_TP=2, ACTOR_CP=2, ACTOR_PP=1
Rollout shape: ROLLOUT_TP=1
```

Start from `launch_swe_qwen35_9b_megatron_b200.yaml` and set full-training
environment values like this:

```yaml
arnold:
  region: i18n
  group_name: mlsys_inference
  cluster_name: cloudnative-useast1b
  quota_pool: third_party_oci
  image_url: aliyun-va-hub.byted.org/arnold/jason.wei:qwen35b200baseplainenvcodepy4
  job_name: swe_qwen35_9b_megatron_b200x16_full
  namespace: /topic/2ebfba22254a08e7
  use_robust_training: true
  retry_times: 3
  skip_restart: false
  roles:
    - name: head
      num: 1
      cpu: 32
      memory: 65536
      gpuv: CPU_ONLY
      ports: 20
      queue_name: public_CPU
    - name: worker
      num: 2
      gpu: 8
      gpuv: NVIDIA_B200
      ports: 20
      queue_name: compute-598-useast1b-cloudnative-aioci-mlsys.inference-guarantee
  hdfs_mount:
    - hdfs_address: hdfs://harunava/home/byte_arnold_va_ssd/mlsys/models/Qwen3.5-9B
      mount_path: /mnt/hdfs/model_path
      mode: RO
    # Optional but recommended for long runs. Replace with a writable team/user
    # HDFS path and set CHECKPOINT_ROOT to this mount path.
    - hdfs_address: hdfs://harunava/home/<team-or-user>/rllm_swe_checkpoints
      mount_path: /mnt/hdfs/swe_checkpoints
      mode: RW
  envs:
    RLLM_SWE_ARTIFACT_DIR: /opt/tiger/swe_runtime_image/artifacts
    RLLM_SWE_MEGATRON_OVERLAY: /opt/tiger/swe_runtime_image/artifacts/megatron_core_0.18.0_829a7b78d.tar.gz
    RLLM_SWE_OUTPUT_DIR: /tmp/rllm_swe_outputs
    CHECKPOINT_ROOT: /mnt/hdfs/swe_checkpoints
    MODEL_PATH: /mnt/hdfs/model_path
    MODEL_USE_REMOVE_PADDING: "true"
    RLLM_SWE_MODE: full
    RLLM_SWE_REQUIRE_WANDB: "1"
    RLLM_SWE_REQUIRE_MEGATRON_CP2: "1"
    RLLM_SWE_MODAL_PROBE_TOTAL: "12"
    RLLM_SWE_MODAL_PROBE_CONCURRENCY: "6"
    ARNOLD_NETWORK_TOPOLOGY: SAME_NETMINIPOD
    LOGGER: "[console,wandb]"
    WANDB_MODE: online
    NNODES: "2"
    NGPUS_PER_NODE: "8"
    ACTOR_TP: "2"
    ACTOR_CP: "2"
    ACTOR_PP: "1"
    ROLLOUT_TP: "1"
    ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU: "1"
    ACTOR_LR_WARMUP_STEPS: "20"
    SWE_AGENT_TIMEOUT: "600"
    SWE_COMMAND_TIMEOUT: "120"
    SWE_SANDBOX_TIMEOUT: "800"
    SWE_STARTUP_JITTER_S: "25.0"
    SWE_N_PARALLEL_TASKS: "128"
    ROLLOUT_CORRECTION_IS: "null"
    RAY_BACKEND_LOG_LEVEL: info
    RAY_DEDUP_LOGS: "0"
    RAY_LOG_TO_DRIVER: "1"
    NCCL_DEBUG: INFO
    RLLM_SWE_FULL_EXTRA_ARGS: >-
      ++trainer.total_training_steps=100
      trainer.save_freq=10
      trainer.test_freq=1000
      trainer.val_before_train=false
      ++rllm.trainer.val_before_train=false
      actor_rollout_ref.model.use_remove_padding=true
      rllm.workflow.n_parallel_tasks=128
      actor_rollout_ref.rollout.n=8
      rllm.algorithm.rollout_correction.tis_mode=null
      algorithm.rollout_correction.rollout_is=null
      trainer.project_name=swe-verl-9b-megatron
      trainer.experiment_name=swe_qwen35_9b_megatron_b200x16_full
  local_envs:
    - MODAL_TOKEN_ID
    - MODAL_TOKEN_SECRET
    - WANDB_API_KEY

verl:
  entrypoint: submodules/rllm/cookbooks/swe/launchers/arnold_entrypoint_swe_qwen35_9b_megatron.sh
  ray_runtime_env: submodules/rllm/cookbooks/swe/launchers/ray_runtime_env_swe.yaml
```

Launch from the ModelChef repo root:

```bash
cd /path/to/modelchef
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
export WANDB_API_KEY=...

python verl-recipes/tasks/arnold_launch.py \
  --config submodules/rllm/cookbooks/swe/launchers/launch_swe_qwen35_9b_megatron_b200.yaml
```

The entrypoint runs the Modal/SWE-ReX probe on the Arnold-selected CPU head
before starting training. A CPU is accepted only when the dynamic probe passes
with 12/12 successes, no transient failures, and `create_env_s.p95 <= 30s`. If
the selected CPU is bad, the job exits early with `BAD_CPU_DRIVER`; relaunch so
Arnold selects a different CPU. Do not spend B200 time on a CPU that fails this
gate.

For fast iteration, keep the same YAML but reduce the Arnold worker and trainer
shape together:

```yaml
roles:
  - name: worker
    num: 1
    gpu: 4
    gpuv: NVIDIA_B200
envs:
  NNODES: "1"
  NGPUS_PER_NODE: "4"
```

For a one-step smoke test, keep `RLLM_SWE_MODE=smoke` or override the full args
to a small workload:

```yaml
envs:
  RLLM_SWE_MODE: smoke
  RLLM_SWE_SMOKE_STEPS: "1"
  RLLM_SWE_SMOKE_TRAIN_SAMPLES: "16"
  RLLM_SWE_SMOKE_VAL_SAMPLES: "4"
  RLLM_SWE_SMOKE_ROLLOUT_N: "1"
  RLLM_SWE_SMOKE_PARALLEL_TASKS: "4"
```

For long runs, set `trainer.save_freq` to a small value such as `5` or `10` and
set `CHECKPOINT_ROOT` to a writable HDFS mount. Avoid relying on `/tmp` as the
only copy of training state for multi-hour jobs.

To mirror saved checkpoints to Hugging Face, enable the explicit HF upload
switch and pass `HF_TOKEN` through Arnold. The upload runs after each local
checkpoint save and writes the full `global_step_<N>` checkpoint folder to the
repo under `checkpoints/<experiment_name>/global_step_<N>`. HDFS should still be
the primary durable checkpoint sink; HF upload is just a simple mirror.

```yaml
envs:
  HF_UPLOAD_CHECKPOINTS: "true"
  HF_UPLOAD_REPO_ID: JWei05/qwen35-9b-swe-smith-megatron
local_envs:
  - HF_TOKEN
```

Equivalent Hydra overrides:

```yaml
RLLM_SWE_FULL_EXTRA_ARGS: >-
  ++trainer.hf_upload=true
  +trainer.hf_repo_id=JWei05/qwen35-9b-swe-smith-megatron
```

#### Changing Experiment Settings

Do not edit `run_swe_training_9b_megatron.sh` for normal experiment variants.
Change these in the Arnold YAML instead:

```yaml
envs:
  NNODES: "2"
  NGPUS_PER_NODE: "8"
  ACTOR_TP: "2"
  ACTOR_CP: "2"
  SWE_N_PARALLEL_TASKS: "128"
  SWE_AGENT_TIMEOUT: "600"
  SWE_SANDBOX_TIMEOUT: "800"
  RLLM_SWE_FULL_EXTRA_ARGS: >-
    ++trainer.total_training_steps=100
    trainer.save_freq=10
    trainer.experiment_name=my_experiment
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.actor.optim.lr=1e-6
```

Only change the training script when a setting should become a new shared
default or when new environment-variable plumbing is needed.

#### External Ray Worker Diagnostics

The external-Ray-worker config is still useful when debugging CPU placement or
Ray behavior, but it is not the preferred end-state for ByteDance Arnold runs.
Use it when you already have a known-good CPU Ray head and want Arnold to attach
B200 workers to that head:

```bash
cd /path/to/modelchef
export EXTERNAL_RAY_HEAD_IP=REPLACE_WITH_MODAL_GOOD_CPU_IPV6

python verl-recipes/tasks/arnold_launch.py \
  --config submodules/rllm/cookbooks/swe/launchers/launch_swe_qwen35_9b_megatron_b200_external_ray_workers.yaml
```

The manually managed CPU head must still pass the Modal probe described above
before attaching B200 workers.

## Tests

```bash
pytest cookbooks/swe/tests -q
bash -n cookbooks/swe/scripts/setup_verl_vllm018_qwen35.sh
bash -n cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron.sh
bash -n cookbooks/swe/swe/training_scripts/run_swe_training_9b_megatron_h100.sh
```

The pytest suite is unit-style and avoids live Modal grading. Dataset grading
smoke tests should be run manually with small slices when Modal credentials are
available.

## Files

| Path | Description |
|---|---|
| `swe/agent_flow.py` | `SWEAgentFlow` and `ProgressLoggingAgent` rollout logic |
| `swe/openai_model.py` | OpenAI-compatible mini-swe-agent model wrapper |
| `swe/evaluator.py` | Dataset-specific grading router |
| `swe/prepare_data.py` | Converts supported datasets into rollout task JSON |
| `swe/prepare_rllm_data.py` | Registers supported datasets in rLLM `DatasetRegistry` |
| `swe/scripts/run_eval.py` | Single-pass evaluation entry point |
| `swe/scripts/run_n_eval.py` | Multi-sample pass@k evaluation entry point |
| `swe/scripts/train_swe_verl.py` | veRL training entry point |
| `swe/scripts/prepare_filtered_mix.py` | Builds the SWE-smith filtered training mix |
| `swe/scripts/modal_swerex_reliability_test.py` | Modal/SWE-ReX CPU-driver reliability probe |
| `scripts/setup_verl_vllm018_qwen35.sh` | Builds the Qwen3.5 veRL/vLLM training venv |
| `launchers/launch_swe_qwen35_9b_megatron_b200_external_ray_workers.yaml` | Arnold B200-worker-only launch config for an external CPU Ray head |
| `launchers/launch_external_cpu_driver_swe_qwen35_9b_megatron.sh` | External CPU-driver launcher for Arnold worker clusters |
| `launchers/arnold_entrypoint_swe_qwen35_9b_megatron.sh` | Arnold entrypoint for smoke, cluster-hold, and external-Ray-worker modes |
| `launchers/swe_artifact_utils.sh` | Restores prebuilt venv, model/data caches, and Megatron CP2 overlay |
| `swe/training_scripts/` | Production 9B Megatron launchers |
| `swe/tasks/` | SWE-bench Pro, SWE-bench Multilingual, SWE-smith, and SWE-rebench V2 graders |
| `swe/config/` | Agent, Modal, compaction, and veRL Hydra config |
| `tests/` | Unit tests for model wrapping, compaction, trajectories, and retry patching |
