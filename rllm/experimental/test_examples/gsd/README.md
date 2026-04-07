# GSD Math Training Example

Reproduces GSD training on DeepScaleR math problems with the Tinker backend.

## Prerequisites

Register the datasets before launching:

```bash
python -m examples.deepscaler.prepare_math_data
```

This creates `deepscaler_math` (train, ~40k problems) and `aime2024` (test, 30 problems) in the rLLM dataset registry.

## Files

- **`math_utils.py`** -- Dataset loading (`prepare_deepscaler_datasets`) and reward function (`gsd_math_reward`: binary 1.0/0.0 via `RewardMathFn` with `\boxed{}` extraction).

- **`test_gsd_math_tinker.py`** -- Hydra entry point. Creates `GsdConfig`, forces `group_size=1`, wires the GSD estimator map and custom transform, then launches `AgentTrainer` with Tinker backend.

## Launch

```bash
bash tmp/gsd/test_gsd_math_tinker.sh
```

The shell script sets Tinker-native config overrides (model, LR, batch sizes, sampling params). See `tmp/gsd/test_gsd_math_tinker.sh` for the full list.
