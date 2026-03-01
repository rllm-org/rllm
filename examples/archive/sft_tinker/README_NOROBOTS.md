# NoRobots SFT Training with Tinker Backend

This example demonstrates supervised fine-tuning (SFT) on the [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) dataset using rLLM with Tinker's hosted GPU service.

This example replicates the tinker-cookbook's `sl_basic.py` recipe with the following configuration:

- **meta-llama/Llama-3.1-8B** as the base model
- **HuggingFaceH4/no_robots** dataset (9,500 train, 500 test conversations)
- **LoRA fine-tuning** with rank 32
- **Batch size 128**, learning rate 2e-4, linear LR schedule

## Setup

### Install Dependencies

```bash
pip install tinker
pip install -e .[dev]  # Install rLLM
```

### Set Tinker API Key

```bash
export TINKER_API_KEY=your_api_key_here
```

Get your API key from the [Tinker console](https://tinker-docs.thinkingmachines.ai/).

## Dataset Preparation

Download and register the NoRobots dataset:

```bash
cd examples/sft_tinker
python prepare_norobots_dataset.py
```

This will:
- Download the HuggingFaceH4/no_robots dataset
- Register it with rLLM's DatasetRegistry
- Enable loading via `DatasetRegistry.load_dataset("norobots", "train")`

The dataset contains high-quality, human-generated conversations:
- **Train**: 9,500 conversations
- **Test**: 500 conversations
- **Format**: Multi-turn user/assistant messages

## Training

Run SFT with the default configuration:

```bash
cd examples/sft_tinker
bash train_norobots_tinker.sh
```

**Configuration Options:**
You can override training parameters via environment variables:

```bash
# Adjust batch size and learning rate
export BATCH_SIZE=64
export LEARNING_RATE=1e-4
bash train_norobots_tinker.sh

# Use a different model
export MODEL_NAME=meta-llama/Llama-3.2-3B
bash train_norobots_tinker.sh

# Override any Hydra parameter
bash train_norobots_tinker.sh \
    model.lora_rank=64 \
    trainer.logger=['console','wandb']
```

The training script will:
- Load the NoRobots dataset from DatasetRegistry
- Fine-tune meta-llama/Llama-3.1-8B with LoRA (rank 32)
- Train for 1 epoch with batch size 128
- Save checkpoints to `/tmp/rllm-tinker-examples/sl_basic`

**Key Training Settings:**
- `model.name`: Base model to fine-tune (default: meta-llama/Llama-3.1-8B)
- `model.lora_rank`: LoRA rank (default: 32)
- `data.train_batch_size`: Batch size (default: 128)
- `data.max_length`: Max sequence length (default: 32768)
- `data.renderer_name`: Chat template (default: llama3)
- `optim.lr`: Learning rate (default: 2e-4)
- `trainer.total_epochs`: Number of epochs (default: 1)

## Python API Usage

You can also run training programmatically:

```python
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
from omegaconf import OmegaConf

# Load datasets from registry
train_dataset = DatasetRegistry.load_dataset("norobots", "train")
test_dataset = DatasetRegistry.load_dataset("norobots", "test")

# Load config
config = OmegaConf.load("rllm/trainer/config/tinker_sft_trainer.yaml")

# Override settings
config.model.name = "meta-llama/Llama-3.1-8B"
config.model.lora_rank = 32
config.data.train_batch_size = 128
config.optim.lr = 2e-4

# Train with Tinker backend
trainer = AgentSFTTrainer(
    config=config,
    train_dataset=train_dataset,
    val_dataset=test_dataset,
    backend="tinker"
)
trainer.train()
```

## Monitoring

### Console Output

Training logs metrics every step:

```
Step 8: train_nll=2.8456, lr=2.00e-04
Validation NLL: 2.7823
Step 16: train_nll=2.7234, lr=1.95e-04
```

### Weights & Biases

Enable W&B logging:

```bash
bash train_norobots_tinker.sh \
    trainer.logger=['console','wandb'] \
    trainer.project_name=norobots-sft
```

## Expected Results

With default configuration:
- **Training Time**: ~30-60 minutes
- **Final Train NLL**: ~2.5-3.0
- **Final Val NLL**: ~2.6-3.1
- **Checkpoint Size**: ~150MB (LoRA adapters)

## Troubleshooting

**Dataset not found:**
```bash
python prepare_norobots_dataset.py
```

**TINKER_API_KEY not set:**
```bash
export TINKER_API_KEY=your_api_key
```

**Out of memory:**
```bash
export BATCH_SIZE=64
export MAX_LENGTH=16384
bash train_norobots_tinker.sh
```

## References

- [Tinker-Cookbook sl_basic.py](https://github.com/thinking-machines/tinker-cookbook/blob/main/tinker_cookbook/recipes/sl_basic.py)
- [HuggingFaceH4/no_robots Dataset](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
