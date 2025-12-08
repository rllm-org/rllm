# 🔬 DeepSWE Reproduction Guide

This guide provides step-by-step instructions to reproduce the DeepSWE Pass@1 results on the SWE-Bench Verified dataset.

## 📋 Prerequisites

The following doc assumes you have

- **GPUs** with sufficient VRAM (recommended: 80GB+ for tensor parallelism)
- **Sufficient disk space** (at least 1TB for models and datasets)


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


## 🤖 Running DeepSWE Inference

### 2. Start the VLLM Server

First, start the VLLM server to serve the DeepCoder model:

```bash
# Start VLLM server with tensor parallelism across 8 GPUs
export MAX_CONTEXT_LEN=65536
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve agentica-org/DeepSWE-Preview \
    --tensor-parallel-size 8 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching
```

> ⚠️ **Important**: Wait for the server to fully load before proceeding to the next step. You should see logs indicating the server is ready to accept requests.

### 3. Run the Agent Evaluation

In a new terminal session, run the DeepSWE agent evaluation:

```bash
# Activate the virtual environment (if in new terminal)
source .venv/bin/activate

# Run the DeepSWE agent on SWE-Bench Verified
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
    --exp_name "$EXP_NAME" \
    --temperature "$TEMP" \
    --max_steps_absolute 100 \
    --backend "docker" \
    --condense_history False \
    --max_reward_calc_time 1200 \
    --max_tokens 65536
```

**Parameter Explanation:**
- `--max_workers 54`: Number of parallel workers for processing, reduce if you hit trajectory time limit errors
- `--k 500`: Number of instances to evaluate (max 500 for SWE-Bench Verified)
- `--temperature 1`: Sampling temperature for model responses
- `--max_steps 40`: Maximum steps per trajectory
- `--max_steps_absolute 100`: Absolute maximum steps limit

> 📊 **Expected Runtime**: This evaluation may take several hours depending on your hardware configuration.

## 📈 Evaluation with SWE-Bench Harness

### 4. Generate Submission File

Convert the trajectory results to SWE-Bench format:

```bash
uv run python src/r2egym/agenthub/trajectory/create_swebench_submission.py \
    --traj_file_path "results/deepswe_32b_agent_swebv_eval_temp_1_run_1/test_results.jsonl" \
    --output_json_path "{SWE-BENCH-DIR}/outputs/swebench_submission.json"
```

> 🔧 **Note**: Replace `{SWE-BENCH-DIR}` with the actual path to your SWE-Bench installation directory.

### 5. Run Official SWE-Bench Evaluation

Navigate to your SWE-Bench directory and run the evaluation harness:

```bash
cd {SWE-BENCH-DIR}
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path outputs/swebench_submission.json \
    --max_workers 32 \
    --run_id swebv \
    --cache_level none
```

## 📊 Results

After completion, you'll find:
- **Evaluation results** in the SWE-Bench output directory
- **Detailed logs** showing pass/fail status for each instance
- **Summary statistics** including overall success rate

> 📝 **Note**: This reproduction guide is based on the DeepSWE methodology. Results may vary depending on hardware configuration and software versions.

