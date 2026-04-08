## Installation
```bash
uv venv --python=3.12
uv pip install -e .[harbor,tinker]
```

## Data
To convert SWE-Smith into Harbor format:
```bash
git clone https://github.com/harbor-framework/harbor.git external/harbor
cd external/harbor/adapters/swesmith
uv run run_adapter.py
```

Next, wrap with rLLM dataset:
```bash
python examples/harbor-swe/prepare_data.py
```

Example evaluation with Harbor through the remote runtime engine:
```bash
python examples/harbor-swe/eval.py
```