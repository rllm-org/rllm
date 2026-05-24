# Sokoban Agent

A multi-turn rLLM AgentFlow that plays Sokoban using the LaMer Sokoban environment logic.

Each dataset row stores deterministic generation parameters such as `seed`, `dim_room`, and `num_boxes`. The flow regenerates the room in process, renders it as a row-numbered text board, asks the model for up to `actions_per_turn` moves, executes those moves, and rewards success when all boxes are on targets.

## Files

| File | Description |
|------|-------------|
| `sokoban_flow.py` | AgentFlow loop and action parsing |
| `sokoban_eval.py` | Evaluator reading `episode.artifacts["won"]` |
| `prepare_sokoban_data.py` | Procedural train/test dataset registration |
| `sokoban_pkg/` | Vendored LaMer Sokoban wrapper and room generator |
| `train.py` | Python API training entrypoint |
| `train_tinker.sh` | Tinker backend script |
| `train_verl.sh` | Verl backend script |
| `test.py` | Focused parser, evaluator, and env smoke tests |

## Install

```bash
uv pip install -e ".[tinker]"
uv pip install --no-deps -e tbmf/sokoban
```

## Dataset

```bash
python3 tbmf/sokoban/prepare_sokoban_data.py
```

Default generation matches the LaMer Sokoban Qwen3-4B prepared data and
experiment setup: `train_size=8`, `test_size=128`, `env_seed=4608`,
`dim_room=6x6`, `num_boxes=2`, `search_depth=100`, `max_steps=30`,
`max_sol_steps=21`, `actions_per_turn=3`, `max_turns=7`, and
`mode=text_with_row_numbers`. LaMer trains with `data.train_batch_size=8`
and validates with `data.val_batch_size=16` over the 128-row test file.

Useful overrides for non-comparable local experiments:

```bash
python3 tbmf/sokoban/prepare_sokoban_data.py \
    --train-size 5000 \
    --test-size 200 \
    --dim-room 6x6 \
    --num-boxes 1
```

## Eval

```bash
rllm eval sokoban \
    --agent sokoban \
    --evaluator sokoban \
    --split test \
    --max-examples 20
```

## Training

```bash
bash tbmf/sokoban/train_tinker.sh
```

or:

```bash
python3 tbmf/sokoban/train.py \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-4B-Instruct-2507 \
    training.group_size=8
```

## Tests

```bash
PYTHONNOUSERSITE=1 python3 -m pytest -p no:capture --import-mode=importlib tbmf/sokoban/test.py -v
```
