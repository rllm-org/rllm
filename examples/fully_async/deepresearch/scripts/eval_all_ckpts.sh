#!/bin/bash

# Script to evaluate all checkpoints from global_step_175 down to global_step_25
# Usage: bash eval_all_ckpts.sh [model_name] [num_gpus]

set -e  # Exit on error

model_name=${1:-Qwen/Qwen3-8B}
num_gpus=${2:-8}
base_ckpt_dir="/path/to/ckpts/deepresearch/8b_stale05_rs"

echo "==================================================="
echo "Evaluating all checkpoints"
echo "==================================================="
echo "Base checkpoint directory: ${base_ckpt_dir}"
echo "Model name: ${model_name}"
echo "Number of GPUs: ${num_gpus}"
echo "==================================================="
echo ""

# Create summary log file
timestamp=$(date +%Y%m%d_%H%M%S)
summary_log="eval_logs/eval_all_ckpts_summary_${timestamp}.log"
mkdir -p eval_logs

echo "Summary log: ${summary_log}"
echo ""

# Initialize summary log
echo "==================================================" > "${summary_log}"
echo "Checkpoint Evaluation Summary" >> "${summary_log}"
echo "Started at: $(date)" >> "${summary_log}"
echo "==================================================" >> "${summary_log}"
echo "" >> "${summary_log}"

# Counter for tracking progress
total_ckpts=0
successful_ckpts=0
failed_ckpts=0

# Evaluate checkpoints from 175 down to 25 (in steps of 25)
for step in 200 175 150 125 100 75 50 25; do
    ckpt_path="${base_ckpt_dir}/global_step_${step}/merged_hf"

    # Check if checkpoint exists
    if [ ! -d "${ckpt_path}" ]; then
        echo "⚠️  Checkpoint not found: ${ckpt_path}"
        echo "⚠️  SKIPPED: global_step_${step} (not found)" >> "${summary_log}"
        echo ""
        continue
    fi

    total_ckpts=$((total_ckpts + 1))

    echo "==================================================="
    echo "Evaluating checkpoint ${total_ckpts}: global_step_${step}"
    echo "==================================================="
    echo "Path: ${ckpt_path}"
    echo "Started at: $(date)"
    echo ""

    # Run evaluation
    if bash examples/fully_async/deepresearch/scripts/eval_one_ckpt.sh \
        "${ckpt_path}" \
        "${model_name}" \
        "${num_gpus}"; then

        successful_ckpts=$((successful_ckpts + 1))
        echo "" >> "${summary_log}"
        echo "✓ SUCCESS: global_step_${step}" >> "${summary_log}"
        echo "  Completed at: $(date)" >> "${summary_log}"

        echo ""
        echo "✓ Checkpoint global_step_${step} completed successfully"
        echo ""
    else
        failed_ckpts=$((failed_ckpts + 1))
        echo "" >> "${summary_log}"
        echo "✗ FAILED: global_step_${step}" >> "${summary_log}"
        echo "  Failed at: $(date)" >> "${summary_log}"

        echo ""
        echo "✗ Checkpoint global_step_${step} failed"
        echo ""

        # Ask user if they want to continue
        read -p "Continue with next checkpoint? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping evaluation..."
            break
        fi
    fi

    echo ""
    echo "Progress: ${successful_ckpts}/${total_ckpts} successful, ${failed_ckpts}/${total_ckpts} failed"
    echo ""

    # Brief pause between checkpoints
    sleep 5
done

# Write summary
echo "" >> "${summary_log}"
echo "==================================================" >> "${summary_log}"
echo "Evaluation Summary" >> "${summary_log}"
echo "==================================================" >> "${summary_log}"
echo "Total checkpoints: ${total_ckpts}" >> "${summary_log}"
echo "Successful: ${successful_ckpts}" >> "${summary_log}"
echo "Failed: ${failed_ckpts}" >> "${summary_log}"
echo "Completed at: $(date)" >> "${summary_log}"
echo "==================================================" >> "${summary_log}"

# Print final summary
echo ""
echo "==================================================="
echo "All evaluations completed!"
echo "==================================================="
echo "Total checkpoints: ${total_ckpts}"
echo "Successful: ${successful_ckpts}"
echo "Failed: ${failed_ckpts}"
echo ""
echo "Summary log: ${summary_log}"
echo "Individual logs: eval_logs/eval_global_step_*"
echo "Results: eval_results_browsecomp_*.json"
echo "==================================================="
