# DeepScaleR

To replicate our reported numbers for `DeepScaleR-1.5B-Preview`, run:
```bash
./scripts/eval/eval_model.sh --model agentica-org/DeepScaleR-1.5B-Preview --datasets aime math amc minerva olympiad_bench --output-dir $HOME/DeepScaleR-1.5B-Preview --tp 1 --n 16 --max-length 32768
```

# DeepCoder

## LiveCodeBench

```bash
./scripts/eval/eval_model.sh --model agentica-org/DeepCoder-14B-Preview --datasets test_livecodebench --output-dir $HOME/DeepCoder-14B-Preview --tp 4 --max-length 65536
```

## Codeforces

Run `eval_model.sh` to generate `results.json`, which is used to calculate Codeforces ELO and percentile:

```bash
./scripts/eval/eval_model.sh --model agentica-org/DeepCoder-14B-Preview --datasets test_codeforces --output-dir $HOME/DeepCoder-14B-Preview --tp 4 --max-length 65536 --n 8
```

Then, in `scripts/deepcoder/benchmark`, run:

```python
python scripts/deepcoder/benchmark/cf_elo_calc.py --results_path [RESULTS_JSON_PATH] --pass_n 8
```