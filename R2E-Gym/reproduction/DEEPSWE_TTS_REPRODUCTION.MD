# 🔬 DeepSWE Test-Time Scaling (TTS) Reproduction Guide

This guide provides step-by-step instructions to reproduce the DeepSWE Test-Time Scaling results on the SWE-Bench Verified dataset. TTS uses multiple rollouts combined with execution-free and execution-based verifiers to achieve higher performance.

## 📋 Prerequisites

The following doc assumes you have

- **GPUs** with sufficient VRAM (recommended: 80GB+ for tensor parallelism)
- **Sufficient disk space** (at least 1TB for models and datasets)

> ⚠️ **Note**: TTS requires more computational resources than single-pass evaluation due to multiple rollouts and verifier inference. We recommend distributing the work across multiple machines if possible.

## 🚀 Setup Instructions

### 1. Clone Repository and Install Dependencies

```bash
# Clone the R2E-Gym repository
git clone git@github.com:agentica-project/R2E-Gym.git
cd R2E-Gym

# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync && uv pip install -e .
```


## 🤖 Running DeepSWE Test-Time Scaling

### 2. Generate Multiple Agent Rollouts

First, start the VLLM server for the main agent model:

```bash
# Set context length and start VLLM server
export MAX_CONTEXT_LEN=65536
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve agentica-org/DeepSWE-Preview \
    --tensor-parallel-size 8 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching
```

> ⚠️ **Important**: Wait for the server to fully load before proceeding. The prefix caching helps with efficiency across multiple rollouts.

Now generate multiple rollouts (we recommend using ≥ 8 rollouts):

```bash
# Set experiment parameters
export EXP_NAME="deepswe_tts_swebv_eval"
export TEMP=1

# Generate multiple rollouts for test-time scaling
for run_idx in {1..16}; do
    echo "Starting rollout $run_idx/16..."
    
    time python src/r2egym/agenthub/run/edit.py runagent_multiple \
        --traj_dir "./traj" \
        --max_workers 48 \
        --start_idx 0 \
        --k 500 \
        --dataset "R2E-Gym/SWE-Bench-Verified" \
        --split "test" \
        --llm_name "openai/agentica-org/DeepSWE-Preview" \
        --scaffold "r2egym" \
        --use_fn_calling False \
        --exp_name "${EXP_NAME}_run_${run_idx}" \
        --temperature "$TEMP" \
        --max_steps_absolute 100 \
        --backend "docker" \
        --condense_history False \
        --max_reward_calc_time 1200 \
        --max_tokens 65536
        
    echo "Completed rollout $run_idx/16"
done
```

> �� **Expected Runtime**: Each rollout may take a few hours. 

### 3. Run Execution-Free (EF) Verifier

Stop the previous VLLM server and start the verifier model:

```bash
# Stop previous server and start verifier model
export MAX_CONTEXT_LEN=76800
vllm serve Qwen/Qwen3-14B \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable-lora \
    --lora-modules verifier=agentica-org/DeepSWE-Preview \
    --port 8000 \
    --dtype bfloat16 \
    --max-lora-rank 64 \
    --tensor-parallel-size 8
```

> ⚠️ **Important**: Wait for the LoRA adapter to load completely before proceeding.

Run the execution-free verifier on all rollouts:

```bash
# Run execution-free verifier
uv run python src/r2egym/agenthub/verifiers/run_ef_verifier.py \
    --traj_file_glob "traj/${EXP_NAME}_run_*.jsonl" \
    --verifier_model_name "openai/agentica-org/DeepSWE-Preview"
```

**What this does:**
- Analyzes trajectory quality using a specialized verifier model
- Uses the verifier probability to score each trajectory


> ⏱️ **Expected Runtime**: 1-3 hours depending on the number of rollouts.

### 4. Run Execution-Based (EB) Verifier

Run the execution-based verifier for ground-truth validation:

```bash
# Run execution-based verifier
uv run python src/r2egym/agenthub/verifiers/run_eb_verifier.py \
    --traj_file_glob "traj/${EXP_NAME}_run_*.jsonl" \
    --max_workers 42
```

**What this does:**
- Runs the regression tests on the generated code/patches
- Runs (model generated) reproduction tests on the generated code/patches
- Here, we use reproduction tests from the R2E-TestgenAgent model 

> ⏱️ **Expected Runtime**: 1-3 hours due to actual code execution 

> 🔧 **Note**: Reduce `--max_workers` if you encounter resource constraints or Docker issues.

### 5. Create Best-of-N Verifier Submission

Aggregate the best solutions from multiple rollouts using verifier scores:

```bash
# Create aggregated submission file
uv run python src/r2egym/agenthub/verifiers/create_bestofn_aggregate.py \
    --traj_file_glob "traj/${EXP_NAME}_run_*.jsonl" \
    --verifier_mode "hybrid" \
    --output_json_path "{SWE-BENCH-DIR}/outputs/swebench_submission_tts.json"
```

**Verifier Modes:**
- `"hybrid"`: Combines both EF and EB verifier scores (recommended)
- `"ef"`: Uses only execution-free verifier scores
- `"eb"`: Uses only execution-based verifier scores

> 🔧 **Note**: Replace `{SWE-BENCH-DIR}` with the actual path to your SWE-Bench installation directory.

## 📈 Evaluation with SWE-Bench Harness

### 6. Run Official SWE-Bench Evaluation

Navigate to your SWE-Bench directory and run the evaluation harness:

```bash
cd {SWE-BENCH-DIR}
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path outputs/swebench_submission_tts.json \
    --max_workers 32 \
    --run_id swebv_tts \
    --cache_level none
```
