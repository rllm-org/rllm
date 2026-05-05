## Harbor SWE Example

Train and evaluate SWE agents on Harbor tasks using the rLLM CLI.

### Installation
```bash
uv venv --python=3.12
uv pip install -e .[harbor,tinker]
```

### Evaluation
Evaluate on SWE-Bench Verified (dataset is auto-pulled on first run):
```bash
rllm eval harbor:swebench-verified --agent harbor:mini-swe-agent
```

### Training
Fully-async RL training with the tinker backend on SWE-Smith:
```bash
bash examples/harbor_swe/train_harbor.sh
```

The script automatically pulls the `harbor:swesmith` and `harbor:swebench-verified` datasets before launching training. Edit `train_harbor.sh` to adjust the model, hyperparameters, or W&B project name.
