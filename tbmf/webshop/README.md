# WebShop Agent

A multi-turn rLLM AgentFlow for the WebShop shopping simulator.

Each dataset row stores a deterministic session id plus the environment
configuration used by the rollout. The flow builds the source WebShop prompt,
lets the model emit one `search[...]` or `click[...]` action per turn, and
scores the episode from `episode.artifacts["won"]`.

## Files

| File | Description |
|------|-------------|
| `prepare_webshop_data.py` | Registers train/test WebShop session rows |
| `webshop_flow.py` | AgentFlow loop, prompt builder, and action parsing |
| `webshop_eval.py` | Sparse success evaluator |
| `train.py` | Python API training entrypoint |
| `test.py` | Parser, prompt, dataset, and flow smoke tests |

## Data

The WebShop package resolves data from `datasets/webshop/webshop_data`
or an exported `WEBSHOP_DATA_ROOT`.
The environment also needs Lucene search indexes under
`rllm/rllm/environments/webshop/webshop_pkg/search_engine/indexes_1k/` for the
default `num_products=1000` setting.

Prepare the registry:

```bash
python3 tbmf/webshop/prepare_webshop_data.py
```

## Training

```bash
python3 -m tbmf.webshop.train rllm/backend=tinker
```

## Tests

```bash
PYTHONNOUSERSITE=1 python3 -m pytest -p no:capture --import-mode=importlib tbmf/webshop/test.py -v
```
