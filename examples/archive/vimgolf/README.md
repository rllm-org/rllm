# VimGolf Training Examples

This directory contains examples for training of VimGolf agent models using the RLLM framework. The VimGolf agent training pipeline uses 612 VimGolf public challenges and VimGolf validator for checking agent solutions.

You need to have Vim installed on local machine.

Our examples use the following:

- **deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B** as the base model
- **VimGolf Public Challenges** for training data

## Dataset Preparation

First prepare the dataset:

```bash
cd examples/vimgolf
python prepare_vimgolf_data.py
```

This will generate a dataset named `vimgolf-public-challenges` at `DatasetRegistry`.


## Model Hosting

### Option 1: Using vLLM

Start a vLLM server with OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "<model_saved_path>" \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 
```

### Option 2: Using SGLang

```bash
python -m sglang_router.launch_server \
    --model-path "<model_saved_path>" \ 
    --dp-size 1 \
    --dtype bfloat16
# increase dp_size to enable data-parallel processing on multi-GPU 
```

The server should be accessible at `http://localhost:30000/v1`

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training with the `vimgolf-public-challenges` dataset:

```bash
bash train_vimgolf_agent.sh
```

**Configuration Options:**
You can modify the training script parameters:
- `actor_rollout_ref.model.path`: Base model to train
- `trainer.total_epochs`: Number of training epochs
- `data.train_batch_size`: Total batch size across all GPUs
- `data.micro_batch_size_per_gpu`: Batch size per GPU
- `data.max_prompt_length`: Maximum prompt length
- `data.max_response_length`: Maximum response length

The training script will:
- Load the base model (deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- Fine-tune on the `vimgolf-public-challenges` dataset
- Save checkpoints to `checkpoints/${trainer.project_name}/${trainer.experiment_name}`

## Evaluation

You have to host the trained model at `http://localhost:30000/v1` first before evaluation.

Evaluate the trained model using the saved checkpoint:

```bash
cd examples/vimgolf
python run_vimgolf.py --model_name "<model_saved_path>"
```

Replace `<model_saved_path>` with the actual path to your trained model checkpoint, usually at `checkpoints/${trainer.project_name}/${trainer.experiment_name}`.