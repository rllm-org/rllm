# MiniSweeper Agent

A multi-turn rLLM AgentFlow that plays a compact Minesweeper variant using the LaMer MiniSweeper game engine.

Each dataset row stores deterministic generation parameters such as `seed`, `board_size`, and `n_mines`. The flow regenerates a board in process, renders the visible board as text, asks the model for one coordinate to reveal, and rewards success when all non-mine cells are cleared.

## Files

| File | Description |
|------|-------------|
| `minisweeper_flow.py` | AgentFlow loop and coordinate parsing |
| `minisweeper_eval.py` | Evaluator reading `episode.artifacts["won"]` |
| `prepare_minisweeper_data.py` | Procedural train/test dataset registration |
| `minisweeper_pkg/` | Vendored LaMer MiniSweeper game core |
| `train.py` | Python API training entrypoint |
| `train_tinker.sh` | Tinker backend script |
| `train_verl.sh` | Verl backend script |
| `test.py` | Focused parser, evaluator, and env smoke tests |

## Install

```bash
uv pip install -e ".[tinker]"
uv pip install --no-deps -e tbmf/minisweeper
```

## Dataset

```bash
python3 tbmf/minisweeper/prepare_minisweeper_data.py
```

Default generation matches the LaMer Minesweeper Qwen3-4B prepared data and
experiment setup: `train_size=8`, `test_size=128`, `env_seed=0`,
`board_size=6`, `n_mines=3`, `board_type=board`, `mode=text`,
`max_steps=15`, and `max_turns=7`. LaMer trains with
`data.train_batch_size=8` and validates with `data.val_batch_size=16` over
the 128-row test file.

Useful overrides for non-comparable local experiments:

```bash
python3 tbmf/minisweeper/prepare_minisweeper_data.py \
    --train-size 5000 \
    --test-size 200 \
    --board-size 5 \
    --n-mines 5
```

## Eval

```bash
rllm eval minisweeper \
    --agent minisweeper \
    --evaluator minisweeper \
    --split test \
    --max-examples 20
```

## Training

```bash
bash tbmf/minisweeper/train_tinker.sh
```

or:

```bash
python3 tbmf/minisweeper/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    training.group_size=8
```

## Tests

```bash
PYTHONNOUSERSITE=1 python3 -m pytest -p no:capture --import-mode=importlib tbmf/minisweeper/test.py -v
```
