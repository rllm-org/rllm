set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime")
OUTPUT_DIR="$HOME"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$HOME/rllm/data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.prompt_length=2048 \
        rollout.response_length=32768 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.n_val=1 \
        rollout.gpu_memory_utilization=0.8 \
        rollout.tensor_model_parallel_size=1
done
