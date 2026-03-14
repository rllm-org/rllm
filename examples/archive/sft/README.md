# Math SFT Training Examples

This directory contains examples for supervised fine-tuning (SFT) of math reasoning models using the RLLM framework. The SFT training pipeline generates high-quality trajectories from a teacher model and fine-tunes a student model on the successful trajectories.

Our examples use the following:

- **Qwen/Qwen2.5-Math-7B-Instruct** as the base model
- **Qwen/Qwen3-4B** as the teacher model for trajectory generation
- **DeepScaleR math dataset** for training data

## Model Hosting

### For Data Generation (Teacher Model)
Start a vLLM server with OpenAI-compatible API:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --host 0.0.0.0 \
    --port 30000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 32768
```

The server should be accessible at http://localhost:30000/v1

## Dataset Preparation

First prepare the dataset:

```bash
cd examples/sft
python prepare_math_data.py
```

Generate SFT training data from teacher model trajectories:

```bash
cd examples/sft
python generate_sft_data.py --num_samples 1000 --trajectories_per_problem 4 --reward_threshold 1.0 --output large_sft_data.parquet
```

This will:
- Load problems from the DeepScaleR math dataset
- Generate trajectories using the teacher model (Qwen3-4B)
- Filter trajectories by reward threshold
- Save successful trajectories in SFT format

**Configuration Options:**
- `--num_samples`: Number of problems to generate trajectories for (default: 500)
- `--trajectories_per_problem`: Number of trajectories per problem (default: 4)
- `--reward_threshold`: Minimum reward score to include trajectory (default: 1.0)
- `--output`: Output file name (default: "sft_data.parquet")

## Training

Run SFT with the generated data:

```bash
bash train_math_sft.sh
```

**Configuration Options:**
You can modify the training script parameters:
- `model.partial_pretrain`: Base model to fine-tune
- `trainer.total_epochs`: Number of training epochs
- `data.train_batch_size`: Total batch size across all GPUs
- `data.micro_batch_size_per_gpu`: Batch size per GPU
- `data.max_length`: Maximum sequence length
- `data.train_files`: Training data file
- `data.val_files`: Validation data file

The training script will:
- Load the base model (Qwen2.5-Math-7B-Instruct)
- Fine-tune on the generated SFT data
- Save checkpoints to `outputs/qwen2.5_math_sft/`

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang_router.launch_server \
    --model-path Qwen/Qwen2.5-Math-7B-Instruct \
    --lora-path examples/sft/outputs/qwen2.5_math_sft/global_step_2784 \
    --dp-size 1 \
    --dtype bfloat16 \
    --disable-radix-cache \
    --context-length 16384
    --port 30000
```

The server should be accessible at http://localhost:30000/v1

Evaluate the trained model using the saved checkpoint:

```bash
cd examples/sft
python run_sft_model.py --model_path outputs/qwen2.5_math_sft/global_step_2784/
```

Replace `outputs/qwen2.5_math_sft/global_step_2784/` with the actual path to your trained model checkpoint.





