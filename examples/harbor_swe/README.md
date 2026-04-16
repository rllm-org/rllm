## Installation
```bash
uv venv --python=3.12
uv pip install -e .[harbor,tinker]
```

## Data
To convert SWE-Smith (training) and SWE-Bench Verified (evaluation) into Harbor format:
```bash
git clone https://github.com/harbor-framework/harbor.git external/harbor

cd external/harbor/adapters/swesmith
uv run run_adapter.py

cd external/harbor/adapters/swebench
uv run run_adapter.py --task-dir ../../datasets/swebench_verified
```

Next, wrap both with rLLM dataset:
```bash
python examples/harbor-swe/prepare_data.py
```

## Evaluation
Example evaluation with Harbor on SWE-Bench Verified through the remote runtime engine:
```bash
python examples/harbor-swe/eval.py
```

## Training
To run fully-async trining with the tinker backend, run:
```bash
bash examples/harbor_swe/train_harbor.sh
```