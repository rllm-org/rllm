# FinQA GRPO — Experiment Audit Log

**Objective:** get the rLLM GRPO training loop working end-to-end on the FinQA cookbook, staying as close as possible to `rllm-project.github.io/posts/finqa.md`, then run one full epoch over 4030 tasks with a checkpoint every ~1000 tasks.

**Last updated:** 2026-08-28 15:00 EDT — training at step 68/1250 on .16 (pid 2326140, auto-resumed from global_step_62); judge switched to gpt-5.4-nano; checkpoint evaluation found to be **below its own noise floor** under sampling, so greedy decoding is now the evaluation default.

---

## 1. Current status

```
| Item                    | Status                                                    |
|-------------------------|-----------------------------------------------------------|
| link bring-up           | DONE - full GRPO loop verified end-to-end                 |
| .16 environment         | DONE - driver 580.178.04, full stack verified             |
| finqa.md alignment      | DONE - see section 5                                      |
| stability run (20 steps)| DONE - all 7 acceptance checks PASSED                     |
| checkpoint write        | VERIFIED - global_step_31 and _62 written and merged      |
| validation loop         | VERIFIED - test_freq=125 now lands on epoch boundaries    |
| base line, in-training  | val pass@1 = 0.6226 (temp 0.6/0.95, comparable to finqa.md)|
| judge                   | gpt-5.4-nano, 10 retries on abnormal finish, alarm on fail|
| full training           | RUNNING on .16, step 68/1250, pid 2326140                 |
| checkpoint eval         | INVALID under sampling - 4.21 pt noise on an identical    |
|                         | model; re-running under greedy (see section 10j)          |
```

Progress: step 68 of 1250 = 5.4% (68/1250) of the ten-epoch run, or 54.4% (68/125) of epoch 1. Epoch-boundary validation fires at step 125.

---

## 2. Hardware and roles

```
| Host | Alias | GPUs                   | Driver     | Role                      |
|------|-------|------------------------|------------|---------------------------|
| .29  | -     | 2x RTX 5000 Ada 32GB   | 580.159.03 | judge server (:1702)      |
| .16  | tai   | 2x RTX 6000 Ada 48GB   | 580.178.04 | training                  |
```

Both GPUs are **sm_89**, so the flash-attn build (compiled with `FLASH_ATTN_CUDA_ARCHS=89`) is reusable across both machines without a rebuild.

`.16` mounts `~/agents` from `.29` over NFS (`10.225.68.29:/home/yanan/agents`), so the repo, the venv (13 GB) and this run directory are shared. Only the Python interpreter, the HF model cache and `~/.rllm` are machine-local.

Judge calls go over the network from `.16` to `.29:1702`, so training and judging never contend for the same GPU.

---

## 3. Repository changes

**Total: 2 lines, 1 file.** `verl` repo untouched (`git status` clean).

```diff
--- a/cookbooks/finqa/finqa_eval.py
+++ b/cookbooks/finqa/finqa_eval.py
 OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
-JUDGE_MODEL = "gpt-5-nano"
-MULTI_TABLE_JUDGE_MODEL = "gpt-5-mini"
+JUDGE_MODEL = os.environ.get("FINQA_JUDGE_MODEL", "gpt-5-nano")
+MULTI_TABLE_JUDGE_MODEL = os.environ.get("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5-mini")
```

Purely additive — defaults unchanged. Everything else (LoRA keys, sequence lengths, memory, CUDA runtime) was solved through hydra overrides and environment variables, never by editing the repo.

`finqa_flow.py`, `finqa_tools.py`, `train.py`, `train_verl.sh`, `finqa_constants.py` are all untouched. `AgentTrainer`, `finqa_flow`, `finqa_evaluator` and the unified hydra config are reused as shipped.

---

## 4. Files

```
| Path                                        | Role                              |
|---------------------------------------------|-----------------------------------|
| ~/agents/finqa-grpo-run/train_16.sh         | main script: smoke/stability/literal/epoch |
| ~/agents/finqa-grpo-run/train_finqa_subset.py| = cookbook train.py + random subsampling  |
| ~/agents/finqa-grpo-run/env.sh              | judge endpoint, LD_LIBRARY_PATH, paths     |
| ~/agents/finqa-grpo-run/preflight_judge.py  | judge fail-fast check                      |
| ~/agents/finqa-grpo-run/register_multi_table.py| registers multi_test/val/train splits   |
| ~/agents/finqa-grpo-run/eval_full.sh        | post-training eval on val/test/multi_test  |
| ~/agents/finqa-grpo-run/setup_16.sh         | one-time .16 setup                         |
| ~/agents/finqa-grpo-run/vllm_probe.py       | real-generation sanity check (137*24=3288) |
| ~/agents/finqa-grpo-run/logs/               | all run logs                               |
| ~/agents/finqa-grpo-run/checkpoints/        | LoRA checkpoints (empty until step 10)     |
| ~/agents/rllm/cookbooks/finqa/data/         | dataset, 55 MB, on NFS                     |
```

---

## 5. Alignment with finqa.md

```
| Parameter                | finqa.md      | ours          | Match | Note                          |
|--------------------------|---------------|---------------|-------|-------------------------------|
| base model               | Qwen3-4B-2507 | same          | YES   |                               |
| algorithm                | GRPO          | GRPO          | YES   |                               |
| learning rate            | 1e-6          | 1e-6          | YES   |                               |
| rollouts per prompt (n)  | 8             | 8             | YES   |                               |
| mini-batch size          | 32            | 32            | YES   | gradient batch 32x8=256 seqs  |
| clip ratio               | 0.28          | 0.28          | YES   |                               |
| entropy coefficient      | 0.002         | 0.002         | YES   |                               |
| KL coefficient           | 0.001         | 0.001         | YES   | pulls in a reference policy   |
| temperature (train)      | 0.7           | 0.7           | YES   |                               |
| temperature (val)        | 0.6           | 0.6           | YES   |                               |
| top-p (val)              | 0.95          | 0.95          | YES   |                               |
| max agent steps          | 20            | 20            | YES   | finqa_flow.py MAX_TURNS=20    |
| training strategy        | FSDP2         | FSDP2         | YES   |                               |
| gradient checkpointing   | Enabled       | Enabled       | YES   |                               |
| prefix caching           | Enabled       | Enabled       | YES   | vLLM v1 default               |
| train batch size         | 256           | 32            | NO    | 2-GPU compute limit           |
| total epochs             | 10            | 1             | NO    | as requested                  |
| validation rollouts (n)  | 8             | 1             | NO    | 8x cost: 522x8=4176 per val   |
| optimizer offload        | Enabled       | Enabled       | YES   | re-enabled, see 6.5           |
| max prompt length        | 2,048         | 8,192         | NO    | different stack, see 6.6      |
| max response length      | 16,384        | 2,048         | NO    | different stack, see 6.6      |
```

On `train_batch_size`: the paper's **gradient** batch is `mini_batch 32 x n 8 = 256 sequences`, and ours is identical. The difference is that they run 8 updates per rollout phase (more off-policy) while we run 1 (fully on-policy). At batch 32 one epoch is **126 steps**, close to the paper's **120 total steps**.

---

## 6. Problems found and resolved

### 6.1 Packaging: five blockers before anything ran

```
| # | Issue                                          | Fix                            |
|---|------------------------------------------------|--------------------------------|
| 1 | tool.uv settings are cwd-sensitive             | run uv from the rllm dir       |
| 2 | verl(numpy<2) vs vllm(numpy>=2) conflict       | rely on override-dependencies  |
| 3 | build isolation injected torch but not numpy   | --no-build-isolation           |
| 4 | g++ 13.3 exceeds CUDA 12.0 host-compiler cap   | CC/CXX/CUDAHOSTCXX = gcc-12    |
| 5 | vllm._C needs libcudart.so.13, torch is cu129  | LD_LIBRARY_PATH -> nvidia/cu13 |
```

Issue 2 is structural: `verl==0.8.0` requires `numpy<2.0.0` while `vllm==0.22.1` pulls `opencv-python-headless>=4.13` which requires `numpy>=2`. rLLM's `tool.uv.override-dependencies = [..., 'numpy>=1.26.0', ...]` exists specifically to break that, and it only applies when uv discovers rLLM's `pyproject.toml` from the cwd.

flash-attn had to be built from source: `vllm==0.22.1` hard-pins `torch==2.11.0`, and flash-attn v2.8.3 (the latest release) ships no prebuilt wheel for torch 2.11. Build took **12m20s** with `FLASH_ATTN_CUDA_ARCHS=89` (default is `80;90;100;120`).

### 6.2 LoRA was silently disabled — the most dangerous bug found

`verl/workers/config/model.py` has **two unconnected LoRA configs**:

```python
    # fsdp lora related
    lora_rank: int = 0            # <- the FSDP engine reads THIS flat key
    lora_alpha: int = 16
    # megatron lora config
    lora: dict[str, Any] = ...    # <- nested dict, Megatron only
```

`verl/workers/engine/fsdp/transformer_impl.py:142` is `self._is_lora = self.model_config.lora_rank > 0`.

The cookbook's shipped `train_verl.sh` sets only `model.lora.rank=32 / lora.alpha=32 / lora.merge=true` — the **Megatron** form. With `strategy=fsdp`, the flat `lora_rank` stayed **0**, so the run was doing **full-parameter fine-tuning** while appearing to train LoRA. (The same repo's `train_tinker.sh` and `train_fireworks.sh` use the correct flat `model.lora_rank`.)

Log evidence:

```
Qwen3ForCausalLM contains 4.02B parameters       <- LoRA NOT applied
PeftModelForCausalLM contains 4.09B parameters   <- LoRA applied (after fix)
```

**Fingerprint of this bug:** OOM lands in `optimizer_step` (not forward/backward), and **changing sequence length does not change the OOM byte count at all**. Two consecutive OOMs reported an identical `28.86 GiB` / `Tried to allocate 194.00 MiB`, which is what led back to LoRA instead of further shrinking sequences.

```
| Metric                    | LoRA broken | LoRA fixed |
|---------------------------|-------------|------------|
| model class               | Qwen3ForCausalLM | PeftModelForCausalLM |
| parameters                | 4.02B       | 4.09B      |
| After FSDP, allocated     | 7.49 GB     | 3.87 GB    |
| outcome                   | OOM         | runs       |
```

The 7.49 -> 3.87 GB drop also confirms `strategy=fsdp2` is genuinely sharding across the two GPUs.

**Danger on larger cards:** on 2x48GB this bug would **not** OOM. It would quietly train something entirely different from what the config claims.

Correct key set (from `verl/examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh`) — both forms are needed:

```
actor_rollout_ref.model.lora_rank=32
actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.model.lora.merge=True     # no + prefix: the key already exists, hydra rejects append
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.rollout.load_format=safetensors
actor_rollout_ref.rollout.layered_summon=True
```

### 6.3 `ppo_max_token_len_per_gpu` cannot be lowered below the longest sequence

`verl/utils/seqlen_balancing.py:384`:

```python
assert max_token_len >= max_seq_len
```

`use_dynamic_bsz` packs **whole sequences** and never splits one, so this parameter is bounded below by `max_prompt_length + max_response_length`. Logits memory is roughly `seq_len x vocab(151936) x ~10 bytes` (bf16 logits + fp32 copy + fp32 grad).

### 6.4 Driver 550 cannot run the CUDA-13 vLLM

`vllm==0.22.1`'s compiled extension is built against CUDA 13, which requires driver >= 580. `.16` had 550.144.03.

`import vllm._C` **succeeded** (dynamic linking resolves) but the first real runtime call failed:

```
RuntimeError: cudaHostGetDevicePointer failed:
CUDA driver version is insufficient for CUDA runtime version
```

No cu12 build of vllm 0.22.1 exists on PyPI or `wheels.vllm.ai`. Resolved by upgrading `.16` to `nvidia-driver-580-open` (580.178.04).

The apt transition needed care: 550 and 580 both claim the virtual packages `nvidia-kernel-common` / `nvidia-kernel-source`, so a plain install fails. The working form marks the old packages for removal in the same transaction:

```bash
sudo apt install -y nvidia-driver-580-open \
    nvidia-driver-550- nvidia-dkms-550- nvidia-kernel-common-550- nvidia-kernel-source-550-
```

That still hit a **file-level** conflict (`libnvidia-extra-550` owns `/usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so`) which `apt-get -s` cannot predict — dependency simulation does not catch file overlaps. Recovery was `dpkg --purge --force-depends` on the leftovers, then `apt --fix-broken install`.

### 6.5 vLLM `wake_up()` OOM after enabling KL

With the KL reference model added and `param_offload=False`, the first `update_weights` died:

```
Call to wake_up method failed: Worker failed with error
'CUDA Error: out of memory at cumem_allocator.cpp:139'
```

In a hybrid engine, vLLM re-allocates its KV cache on wake; the torch caching allocator does not return memory to the driver, so vLLM's `cuMemCreate` fails.

Offload had been turned off on the reasoning that "LoRA's optimizer state is tiny, so offload is pure overhead". **That holds for the optimizer but not for params** — `param_offload` moves the 3.87 GB of base weights off the GPU and empties the torch cache, which is exactly what `wake_up` needs. Measured cost of offload on this box: `update_actor` 164.4s -> 171.5s (~4%).

Fix: `param_offload=True`, `optimizer_offload=True`, `gpu_memory_utilization` 0.50 -> 0.42.

### 6.6 finqa.md's 2048/16384 belong to a different stack

The two stacks define a Step differently:

```
old stack (projects/finqa, AgentExecutionEngine):
    prompt   = initial task description        -> 2,048 is enough
    response = the entire 20-turn trajectory   -> needs 16,384

new stack (cookbooks/finqa, AgentFlow; one Step per LLM call):
    prompt   = the whole conversation so far (grows monotonically)
    response = a single assistant message
```

Measured across four training steps on both machines:

```
| Measurement            | .29 s1 | .29 s2 | .16 s1 | .16 s2 |
|------------------------|--------|--------|--------|--------|
| batch/max_prompt_len   |  2,640 |  2,898 |  2,591 |  2,885 |
| batch/min_prompt_len   |  1,073 |  1,076 |  1,075 |  1,075 |
| batch/mean_response_len|    211 |    219 |    206 |    207 |
| batch/max_response_len |    299 |    371 |    316 |    341 |
```

A dedicated control arm (`./train_16.sh literal`) ran the literal values:

```
| Arm                       | prompt/response | rejections | rate  | steps done |
|---------------------------|-----------------|------------|-------|------------|
| literal (finqa.md verbatim)| 2,048 / 16,384 |        188 | 14.1% | 0 in 15 min|
| ours                      | 8,192 / 2,048   |          0 |  0.0% | running    |
```

Every one of the 188 rejections fired at prompt `value=2049` — one token past 2048:

```
max context 18432; requested 16384 output + at least 2049 input = 18433
```

This is a **hard request rejection**, not a truncate-and-continue. Because the ReAct conversation grows monotonically, once a trajectory crosses 2048 every subsequent turn in it also fails, so the per-trajectory damage far exceeds 14%.

`max_model_len` must also keep headroom rather than being the exact sum: on `.29`, `4096+1024=5120` let a 4097-token prompt trip the same check (0.46%).

### 6.7 Subsampling was taking the head of the dataset, not a random sample

`train_finqa_subset.py` originally used `ds.select(range(n))`. The FinQA CSVs are grouped by company:

```
| Sampling   | distinct companies | out of |
|------------|--------------------|--------|
| first 640  |                 27 |    165 |
| random 640 |                162 |    165 |
```

A head-slice trains on 16% of the table schemas, so any conclusion drawn from it fails to generalise. Now `ds.shuffle(seed=SUBSET_SEED).select(range(n))`, with the realised company count printed to the log for verification. Override the seed via `FINQA_SUBSET_SEED` to repeat with a different draw.

---

## 7. Dataset

```
| Split                | Rows | Registered | Used for                            |
|----------------------|------|------------|-------------------------------------|
| finqa/train          | 4030 | yes        | training                            |
| finqa/val            |  522 | yes        | in-training validation + final eval |
| finqa/test           |  558 | yes        | final eval                          |
| finqa/multi_test     |  131 | yes (added)| held-out generalization probe       |
| finqa/multi_val      |  126 | yes (added)| -                                   |
| finqa/multi_train    |  991 | yes (added)| NOT for training (see below)        |
```

The three primary splits are **100% single-table** (`multi_table` row count is 0), which matches finqa.md's dataset table exactly. Two consequences:

- reward is **naturally binary** (`multi_table = qtype.startswith("multi_table")` is always False), which is the variant that won in the paper's ablation (66.3% binary vs 54.0% partial)
- the judge only ever takes the fast single-table path

`prepare_finqa_data.py` deliberately does not load `data/multi_table_data/`. finqa.md Finding 1 shows adding multi-table data **hurts** (66.3% single-only vs 61.6% mixed vs 64.8% curriculum). The multi-table **test** set is registered separately as a generalization probe only — the paper reports base 13.9% -> trained 26.6% on the equivalent FinQA-Reasoning set.

---

## 8. Judge

```
| Item              | Value                                    |
|-------------------|------------------------------------------|
| model             | Qwen/Qwen3.8-27B-FP8                     |
| endpoint          | http://10.225.68.29:1702/v1              |
| API surface       | Responses API (client.responses.create)  |
| single-table path | reasoning.effort=low, verbosity=low      |
| multi-table path  | effort=medium + json_schema strict       |
```

`finqa_eval._call_judge` uses the **Responses API**, not chat.completions — the endpoint documentation only covers chat.completions, so this was verified explicitly before use.

Preflight results (`preflight_judge.py`):

```
| Check                        | Result | Latency |
|------------------------------|--------|---------|
| single-table, CORRECT answer | True   | 5.1s    |
| single-table, WRONG answer   | False  | 4.8s    |
| multi-table rubric (6 keys)  | 1.000  | 19.9s   |
```

The second row is the important one: it proves the reward signal discriminates. `_call_judge` swallows every exception and returns `0.0`/`False`, so a misconfigured judge would otherwise surface as a plausible-looking 0% score rather than an error.

Measured token usage: **559 input / 34 output** per call, of which ~500 input tokens are the invariant system prompt (89%, fully cacheable).

Judge cost is not a constraint — a full epoch is 32,240 calls ≈ 18.0M input / 1.1M output tokens. Self-hosted, so zero API cost; for reference, at GPT-5.4-mini rates that would be ~$18 uncached, ~$8 cached.

---

## 9. Measurements

### Timing, per step (batch 32, full finqa.md params, `.16`)

```
| Step | timing_s/step | update_actor | old_log_probs | ref (KL) | tokens    | max_mem GiB |
|------|---------------|--------------|---------------|----------|-----------|-------------|
|    1 |       1,352.3 |        721.1 |         229.2 |    206.5 | 3,307,559 |       35.34 |
|    2 |       1,322.1 |        690.9 |         217.6 |    196.5 | 3,130,455 |       35.34 |
|    3 |       1,143.5 |            - |             - |        - |         - |       35.34 |
|    4 |       1,220.2 |            - |             - |        - |         - |       35.34 |
| mean |       1,259.5 |              |               |          |           |       35.34 |
```

Step-time variance is **2.2%** at batch 32, versus **36%** at batch 10 (288s vs 392s on the earlier smoke). Peak allocated memory is byte-identical across all four steps, so the allocator has reached steady state and the remaining 12 GiB of headroom is real.

`update_actor` is 53% of the step. The bottleneck is micro-batch count: `3,307,559 tokens / 2 GPUs / 12288 = 134 micro-batches per GPU`, each a full forward+backward.

`ref` (the KL reference forward) costs ~200s/step = **~7 hours over a full epoch**.

### Cross-machine comparison

```
| Metric                      | .29 (32GB, seq 5120, no KL) | .16 (48GB, seq 12288, full) |
|-----------------------------|-----------------------------|-----------------------------|
| perf/throughput per GPU     |               1,095 tok/s   |               1,184-1,223   |
| perf/max_memory_allocated   |                 13.55 GiB   |                 35.34 GiB   |
| s per trajectory            |                     4.64    |                        4.9  |
| rollout_actor_probs_pearson |                   0.9908    |                     0.9913  |
```

`rollout_actor_probs_pearson_corr ~ 0.991` throughout confirms vLLM (cu13) and the FSDP training side (cu12.9) agree numerically — important because the mixed CUDA runtime is a non-standard combination.

### Learning signal (batch 32, 128 tasks over 4 steps)

```
| Step | reward mean | solve_all | solve_none | solve_partial | fraction_zero | entropy |
|------|-------------|-----------|------------|---------------|---------------|---------|
|    1 |       0.711 |     0.438 |      0.156 |         0.406 |         0.594 |   0.103 |
|    2 |       0.605 |     0.281 |      0.250 |         0.469 |         0.531 |   0.106 |
|    3 |       0.629 |     0.406 |          - |             - |         0.656 |   0.120 |
|    4 |       0.590 |     0.250 |          - |             - |         0.531 |   0.115 |
| mean |       0.634 |     0.344 |            |               |         0.578 |   0.111 |
```

Base reward is ~0.63 on the public HF split. This is far above the 27.9% finqa.md reports for the base model, but that number is measured on Snorkel's **private** benchmark — the public `rLLM/finqa` split is easier, so the two are not comparable and a local base line is required before claiming any gain.

`solve_partial` sits around 0.41-0.47, meaning nearly half the groups contain both successes and failures — that is exactly the signal GRPO needs. `fraction_zero` of 0.53-0.66 is high but not degenerate. Entropy rises slightly (0.103 -> 0.115), so `entropy_coeff=0.002` is doing its job and the policy is not collapsing.

An earlier claim that reward was "saturated at 90%" was based on a single 8-task step and did not survive larger samples.

### Full-epoch projection

```
126 steps x 1,259.5 s = 158,697 s = 44.1 h
+ 6 x 522-task validation           ~ 1.5 h
--------------------------------------------
TOTAL                               ~ 46 h
```

Cross-checked by token count: `32,240 traj x ~12,900 tok/traj / (1,200 x 2 GPU) ~ 44 h`.

---

### Checkpoint write (verified at step 10)

```
| File                              | Size       | Content                     |
|-----------------------------------|------------|-----------------------------|
| model_world_size_2_rank_{0,1}.pt  | 4.16 GB x2 | FSDP2-sharded model state   |
| optim_world_size_2_rank_{0,1}.pt  | 265 MB x2  | optimizer state             |
| extra_state_world_size_2_rank_*.pt| 15 KB x2   | RNG / scheduler             |
| lora_train_meta.json              | 67 B       | {"r":32,"lora_alpha":32}    |
| fsdp_config.json                  | 46 B       |                             |
| huggingface/                      | 11 MB      | config + tokenizer + template|
| latest_checkpointed_iteration.txt | -          | 10                          |
| TOTAL                             | 8.3 GB     |                             |
```

Two independent confirmations that LoRA is genuinely active, beyond the `PeftModelForCausalLM` assertion:

- `lora_train_meta.json` records `r=32, lora_alpha=32` — this file is only written on the peft save path
- optimizer state is **265 MB per rank**. Full-parameter Adam on a 4B model would be in the GB range; 265 MB means the optimizer tracks adapter parameters only

The model shards hold the combined base+adapter state (8.3 GB total = full bf16 weights), not the adapter alone, so the `verl.model_merger` step in `eval_full.sh` must handle this layout. To be confirmed when the first real merge runs.

### Known non-blocking observations

```
| Observation                | Count | Rate   | Assessment                       |
|----------------------------|-------|--------|----------------------------------|
| hermes_tool_parser failures|    34 | 0.12%  | model emits malformed JSON       |
```

`json.decoder.JSONDecodeError: Expecting ',' delimiter` — the parser correctly locates the `<tool_call>` block but the model omitted a comma mid-argument. This is generation quality on long SQL arguments, not a parser misconfiguration: 99.88% of 28,216 calls parse fine, which also confirms `hermes` is the right parser for Qwen3-4B-Instruct-2507 (only Qwen3.5/3.6 need `qwen3_coder`).

From an RL standpoint this is useful signal rather than a defect: malformed JSON means the tool call fails, the trajectory misses the answer, reward is 0, and GRPO pushes that behaviour down. Worth re-checking the rate after training — it should fall.

## 10. Full epoch configuration (as it stands)

```
| Item                       | Value                                       |
|----------------------------|---------------------------------------------|
| model                      | Qwen/Qwen3-4B-Instruct-2507                 |
| judge                      | Qwen3.8-27B-FP8 @ .29:1702                  |
| train / val tasks          | 4030 / 522 (both full)                      |
| train_batch_size           | 32  -> 126 steps per epoch                  |
| rollout.n                  | 8   -> 32,240 trajectories                  |
| ppo_mini_batch_size        | 32                                          |
| save_freq                  | 31 steps = 992 tasks (~ every 1000)         |
| test_freq                  | 31, plus val_before_train and the last step |
| LoRA                       | rank 32 / alpha 32, flat keys               |
| max_prompt / response      | 8192 / 2048                                 |
| rollout.max_model_len      | 12288 (2048 of headroom over the sum)       |
| ppo_max_token_len_per_gpu  | 12288                                       |
| gpu_memory_utilization     | 0.42                                        |
| param / optimizer offload  | both True                                   |
| strategy                   | fsdp2 (actor and ref)                       |
```

Validation fires at step 0 (`val_before_train`), 31, 62, 93, 124 and the final step — `verl` guarantees the last step through `is_last_step` (`ray_trainer.py:1702`).

---

## 10b. Stability run — final acceptance (2026-08-26)

```
| # | Criterion            | Result                                     | Verdict |
|---|----------------------|--------------------------------------------|---------|
| 1 | 20 steps completed   | 20 / 20                                    | PASS    |
| 2 | hard failures        | 0                                          | PASS    |
| 3 | context rejections   | 0                                          | PASS    |
| 4 | checkpoints          | step_10 + step_20, 8.3 GB, lora_meta OK    | PASS    |
| 5 | validation executed  | 2 runs: 0.609 / 0.625                      | PASS    |
| 6 | memory stable        | 35.34 -> 35.99 GiB (+1.9%, no upward trend)| PASS    |
| 7 | no divergence        | entropy 0.103 -> 0.114 rising; reward flat | PASS    |
```

Entropy rising monotonically over 20 steps confirms `entropy_coeff=0.002` is preventing policy collapse.

## 10c. Base line (the anchor for everything)

Measured by `val_before_train` at step 0 of the training run, on the **full 522-task val split**:

```
| Metric                | this run | earlier aborted run |
|-----------------------|----------|---------------------|
| val/finqa/pass@1      |   0.6226 |              0.6054 |
| val/reward/finqa/std  |   0.4847 |              0.4888 |
| val tasks             |      522 |                 522 |
| time/testing          |  386.7 s |             369.8 s |
| steps_used per traj   |     5.04 |                5.09 |
```

The authoritative anchor is **0.6226** — it comes from the run that is actually continuing, so it shares the same process and weight-evolution path as every later validation point.

Validation is **deterministic and reproducible** under this config (`validation_shuffle: False`, seeded rollout): two separate launches produced `0.6226053639846744` to the last digit.

That means an earlier inference of mine was wrong and is retracted here: I attributed the 0.6054 vs 0.6226 gap to temperature-0.6 sampling noise and derived a "1.7 pt empirical noise floor" from n=2. Bit-identical repeats rule that out. The cause of the 0.6054 reading (first launch, `total_epochs=1`, `test_freq=31`) is **not yet identified** — neither of those settings should affect a step-0 validation.

Practical consequence, and it cuts in the useful direction: because validation is reproducible, the points *within this run* carry no sampling noise relative to each other, so a 2-3 point move is real signal rather than something to discount. Judgement therefore rests on the binomial CI, which is evidence-based, not on a fabricated noise figure.


This settles a question that was open all along:

```
| Source          | base pass@1 | Benchmark                   |
|-----------------|-------------|-----------------------------|
| finqa.md        |      27.9%  | Snorkel private benchmark   |
| ours            |      60.5%  | public rLLM/finqa val split |
```

The public split is **much easier** than the private benchmark the paper reports on. Consequences:

- our absolute scores are **not comparable** to finqa.md's 59.7% — different benchmark, different judge
- headroom is 39.5 points, not the paper's 31.8 points of realised gain from a 27.9% floor
- it explains the persistently high `fraction_zero` (0.53-0.66): many groups are unanimously correct because the tasks are not hard for the base model

Decision thresholds against this anchor:

```
| val pass@1 vs base 62.26% | Interpretation                                |
|---------------------------|-----------------------------------------------|
| rising across points      | training is working                           |
| flat                      | one epoch is not moving this dataset          |
| falling across points     | training is degrading the model               |
```

Because validation is reproducible, a consistent direction across successive points is the signal to read; the binomial CI (+/-4.2 pt at n=522) bounds how much weight to put on any single point in absolute terms.

## 10d. Full training run (launched 2026-08-27 10:2x EDT)

```
| Item              | Value                                       |
|-------------------|---------------------------------------------|
| host              | .16 (tai), pid 900575                       |
| total_epochs      | 10  ->  1,260 steps                         |
| test_freq         | 63                                          |
| save_freq         | 31 (= 992 tasks)                            |
| resume_mode       | auto                                        |
| ETA               | ~444 h (~18.5 days) + ~2.1 h validation     |
```

`test_freq` **must divide 126** or epoch boundaries are silently skipped. With the earlier value of 31 the validations land on 31/62/93/124/155 and step 126 is missed entirely; verl's `is_last_step` fallback only fires on the very last step of the whole run (1260), not on intermediate epoch boundaries. 63 hits every boundary (126 = 63x2, 252 = 63x4, ...) plus one mid-epoch point.

Per-validation cost measured at **369.8 s** (522 tasks, n=1) — about 1/3.4 of a training step, because validation has no backward pass, no reference forward and no weight sync. Total validation overhead across 10 epochs is ~2.1 h, i.e. 0.5% of the run.

Post-training evaluation is **deferred by request**: training must not be interrupted. `eval_full.sh` is ready and checkpoints accumulate every 992 tasks, so any checkpoint can be evaluated later.

## 10f. Judge A/B: Qwen3.8-27B vs gpt-5.4-mini (2026-08-27)

Run entirely against external endpoints, so it did not touch the training on .16.

Both judges agree trivially on clearly-right and clearly-wrong answers, so those carry no information. Cases were therefore generated as controlled perturbations of real ground-truth answers, aimed at the surface where judges actually diverge — format tolerance and near misses. 40 dataset rows x 7 categories = 263 comparable cases.

```
| Category    | n  | agree | qwen3.8 pass | gpt-5.4-mini pass | expected |
|-------------|----|-------|--------------|-------------------|----------|
| exact       | 40 |  100% |           40 |                40 | PASS     |
| reformatted | 40 |  100% |           40 |                40 | PASS     |
| rounded     | 23 |   91% |           21 |                23 | PASS     |
| sign_flip   | 40 |  100% |            0 |                 0 | FAIL     |
| magnitude   | 40 |  100% |            0 |                 0 | FAIL     |
| unrelated   | 40 |  100% |            0 |                 0 | FAIL     |
| vague       | 40 |  100% |            0 |                 0 | FAIL     |
| OVERALL     |263 | 99.2% |  38.4% pass  |       39.2% pass  |          |
```

Three things this settles:

- **Format tolerance is identical.** `reformatted` (`-92` -> `-$92.00`, currency symbol and thousands separators added) is 40/40 for both. This was the main worry: FinQA answers legitimately appear as `$14.7M` / `14,700,000` / `14.7`, and a judge that penalises formatting would depress reward systematically and poison the training signal. It does not.
- **Discrimination is identical.** Across the four wrong-answer categories (160 cases) both judges returned False every single time. No "the number looks close enough" leakage.
- **Strictness is within 0.8 pt** (38.4% vs 39.2%), so swapping judges would not shift the reward distribution materially.

The only 2 disagreements, both in `rounded` (one fewer decimal, e.g. 14.71 -> 14.7), have Qwen3.8 stricter and gpt-5.4-mini accepting. That direction is the safe one for training: a slightly stricter judge pushes the agent toward precise numerics rather than teaching it that approximations pass.

**Decision: keep `Qwen3.8-27B-FP8 @ .29:1702`.** Equivalent discrimination, no API cost across ~325,700 calls, ~2% of step wall-clock, and it runs on .29 so it never contends with training on .16.

**Caveat.** These are *synthetic* perturbations of ground-truth strings, not free-form agent output. Real trajectories carry reasoning text, multiple numbers and mixed units, where divergence could exceed 0.8%. Re-run this against saved episodes once real trajectories are available — that is the complete evidence; this is the cheap approximation of it.

Script: `judge_ab.py`. Raw output: `logs/judge_ab.log`.

## 10e. External audit (Codex, 2026-08-27) — findings and disposition

An independent audit raised five points. Each was checked against the code and, where possible, reproduced. Verdicts below; the guiding constraint is that this run is a **reproduction / does-rLLM-work check**, so deviations from finqa.md and edits to the rLLM repo both need to earn their place.

```
| # | Finding                     | Real? | Severity | Disposition                  |
|---|-----------------------------|-------|----------|------------------------------|
| 1 | turn-weighted loss          | YES   | medium   | document, do not change      |
| 2 | TIS / rollout correction off| YES   | low      | do not change; monitor       |
| 3 | eval may score the base     | LIKELY| HIGH     | fix (our script, not rLLM)   |
| 4 | SQL tool is not read-only   | YES   | HIGH     | reproduced; awaiting decision|
| 5 | episodes not saved          | YES   | low      | do not enable                |
```

### 10e.1 Loss is turn-weighted, not trajectory-equal — confirmed, keeping as is

Confirmed from our own logs:

```
| Metric                        | Value       |
|-------------------------------|-------------|
| actor/global_batch_rollouts   | 256         |
| actor/mini_batch_rows         | 1300 - 1358 |
| => sequences per trajectory   | ~5.2        |
| batch/steps_used              | 4.7 - 5.3   |
```

In the AgentFlow stack each LLM call becomes its own training sequence, and `loss_agg_mode=seq-mean-token-mean` weights every sequence equally — so a 10-turn trajectory carries roughly 5x the weight of a 2-turn one. The optimisation objective is therefore turn-weighted GRPO, not the trajectory-equal vanilla form.

Not changing it, for two reasons. First, this is the shipped behaviour of the maintained code path; altering it means editing rLLM's rLLM->verl batch conversion and loss weighting (est. 1-3 days), which is exactly the kind of core change this run is meant to avoid. Second, finqa.md's own numbers likely come from the **old** stack (`projects/finqa`), where the entire 20-turn trajectory was a single 16,384-token sequence — under which `seq-mean-token-mean` *is* trajectory-equal. The semantics changed with the AgentFlow port, not with our configuration.

### 10e.2 TIS off — leaving off

`tis_mode: None`. finqa.md does not use TIS either, so enabling it would be a deviation, not a correction. Our setup is synchronous on-policy (`ppo_epochs=1`, weights synced every step), so there is no stale-policy gap of the kind `realtime_rl.md` addresses — only the vLLM-vs-FSDP numerical difference, measured at `rollout_probs_diff_mean ~ 0.0068`.

The audit's own recommendation was to test it as a **separate arm**, not to mix it into the production curve. Agreed. What we do instead: `training/rollout_probs_diff_*` is already logged every step and is now part of the patrol — divergence there is the trigger to intervene, rather than pre-emptively changing config.

### 10e.3 Post-training eval may silently score the base model — fixing

`verl.model_merger` places LoRA weights under `lora_adapter/` rather than merging them into the root; serving the root directory would then evaluate the **base** model and report it as the trained result. This is the worst failure class here: a plausible-looking number that means nothing.

`eval_full.sh` is our own script, so fixing it touches nothing in the rLLM repo. Two defences go in: handle the adapter explicitly at merge time, and assert that the trained model's output on a fixed prompt **differs** from the base model's — identical output fails the run loudly instead of producing a silent base-model score.

### 10e.4 SQL tool is not read-only — reproduced, and worse than described

`finqa_tools.sql_query` blocks `SELECT *` and requires a filter/aggregate token, but never checks that the statement *is* a SELECT.

The audit described an `UPDATE` path. That path does **not** reproduce: `UPDATE` has no `FROM` clause, so the tool's `FROM|JOIN` table-name rewrite never fires, the external table name does not exist in SQLite, and the statement fails harmlessly.

`DELETE FROM` does reproduce, because it *does* carry a `FROM` clause and therefore gets rewritten to the real internal table name:

```
BEFORE : [{"category":"Right of use assets"}, {"category":"Current liability"}, ...]
DELETE : Error: 'NoneType' object is not iterable      <- all the agent sees
AFTER  : [{"category":"Current liability"}, ...]        <- the row is gone
ROW DELETED: True
```

`_DB_CONN = sqlite3.connect(":memory:")` is process-wide, so one successful `DELETE` corrupts the table store for **every subsequent rollout in that process**, while the agent only receives an opaque `'NoneType' object is not iterable`.

Proposed guard is 4 lines rejecting non-SELECT statements. Agent-visible behaviour barely changes — it gets an error either way — what changes is that the database stops being destructible. It touches the cookbook, not the training framework.

**Status: awaiting the owner's decision.** Recorded here as a known risk either way. No evidence yet that a model has issued such a statement; episodes are not saved, so absence of evidence is not evidence of absence.

### 10e.5 Episodes not saved — not enabling

`rllm.episode_logging.log_episodes=true` is config-only, but at ~322,400 trajectories x ~75 KB that is roughly 24 GB written to NFS during training, with a real chance of slowing the run. The value for a does-it-work check is low. If the motive is evidence for 10e.4, the 4-line guard removes the risk outright and costs nothing to run.

## 10g. Checkpoint evaluation protocol (.29, 2026-08-28)

`.29` was freed when the Qwen3.8 judge was taken offline, so checkpoint evaluation runs there while `.16` keeps training. `eval_parallel.sh` puts one split on each GPU (GPU0 = val 522, GPU1 = test 558).

Environment equality was verified by fingerprint rather than assumed — the code files are md5-identical because `~/agents` is one NFS filesystem, so both hosts read the same bytes:

```
| Component                  | .29 (eval)       | .16 (training)   | Same |
|----------------------------|------------------|------------------|------|
| python / torch / vllm / verl | 3.11.13 / 2.11.0+cu129 / 0.22.1 / 0.8.0 | identical | YES |
| transformers / flash_attn / peft / numpy | 5.5.4 / 2.8.3 / 0.20.0 / 2.4.6 | identical | YES |
| finqa_eval.py md5          | 850babdc05475a95 | 850babdc05475a95 | YES  |
| finqa_flow.py  md5         | 269e9e593c4e2778 | 269e9e593c4e2778 | YES  |
| finqa_tools.py md5         | e7872f9dc54b4bb0 | e7872f9dc54b4bb0 | YES  |
| env.sh md5                 | f10c76638022c94e | f10c76638022c94e | YES  |
| GPU capability             | sm_89            | sm_89            | YES  |
| GPU model                  | RTX 5000 Ada     | RTX 6000 Ada     | no   |
| driver                     | 580.159.03       | 580.178.04       | no   |
```

All three models (base, step_31, step_62) are evaluated on `.29`, so the hardware difference is held constant across the comparison.

### Three protocol defects found and fixed before any number was trusted

```
| # | Defect                                   | Effect                          |
|---|------------------------------------------|---------------------------------|
| 1 | dataset.toml hijacks --split             | a "val" run scored 558 test rows|
| 2 | concurrent materialisation race          | second eval overwrote the first |
| 3 | sampling not pinned                      | 14.9% task-level run-to-run gap |
```

**Defect 1.** `rllm/cli/eval.py:112` redirects to a materialised benchmark whenever `<RLLM_HOME>/datasets/<name>/dataset.toml` exists and `--agent` is set. The local loader then reads the split from that toml (default `test`) and **ignores `--split`**. Fix: give each split an isolated `RLLM_HOME` containing only the registry parquet files — no `dataset.toml`, no `data/`.

A first attempt at this fix was itself wrong: it isolated the path but seeded both copies with `cp -r ~/.rllm/*`, which carried the already-poisoned materialisation across. The `val` run still reported 558. Caught by external review, not by me.

**Defect 3.** Unpinned, the eval inherits the server default (temperature 1.0) rather than the in-training validation's 0.6/0.95. Two runs of the *same* base model over the *same* test rows disagreed on **14.9%** of tasks. Now pinned to `temperature=0.6, top_p=0.95, seed=1234` via `--sampling-params`, which passes unknown keys straight through (`rllm/cli/_sampling.py`, `SamplingConfig.extra`) — there is no `--seed` flag, but the passthrough works and the gateway echoes `{'temperature': 0.6, 'top_p': 0.95, 'seed': 1234}`.

Fixing the seed is a common-random-numbers variance reduction for the base-vs-checkpoint comparison. It is **not** yet demonstrated to make runs bit-reproducible — that requires re-running the same model over the same tasks twice and measuring, which is scheduled after the three evaluations.

### Two rollout failure modes, both from tool-call generation quality

```
| Failure          | Mechanism                                   | Episode | Scoring   |
|------------------|---------------------------------------------|---------|-----------|
| malformed JSON   | vLLM's hermes parser yields NO tool call;    | present | normal    |
|                  | finqa_flow sees empty tool_calls and breaks, |         | failure   |
|                  | treating the raw text as the final answer    |         |           |
| valid JSON, not  | finqa_flow.py:80 calls args.get() on a str   | MISSING | zero      |
| an object        | or list; AttributeError escapes the          |         |           |
|                  | `except JSONDecodeError` guard               |         |           |
```

Reproduced directly: `json.loads` succeeding says nothing about the result being a dict.

```
"{...}"                   -> dict     .get() OK
"mmm_AssetsAndLiabilities"-> str      AttributeError
["3m","t"]                -> list     AttributeError
null                      -> NoneType AttributeError
```

Observed rate in the base test run: 2 of ~222 tasks (~0.9%), tasks 298 and 300.

Note the asymmetry: `fn(**args)` further down *is* wrapped in `except TypeError`, but the earlier `args.get()` on line 80 has no guard.

**Not fixed during this round, deliberately.** The two-line `isinstance(args, dict)` guard would turn a rollout-killing exception into a tool error the model can see — which changes the training environment's feedback semantics. Mixing that into the current curve would make before/after incomparable. It is recorded as a next-protocol change.

### Reporting requirements for the final table

Every model's score must be reported with the denominator and the error count, not accuracy alone:

```
| Model   | val valid/total | test valid/total | rollout exceptions | malformed-JSON rate |
|---------|-----------------|------------------|--------------------|---------------------|
| base    |                 |                  |                    |                     |
| step_31 |                 |                  |                    |                     |
| step_62 |                 |                  |                    |                     |
```

Exceptions count in the denominator and score zero.

One thing this table cannot do: **exception co-location across checkpoints does not validate the seed.** A changed policy produces different actions from identical random draws, so overlap or divergence there reflects task difficulty x policy version, nothing about RNG reproducibility.

### Agreed sequence

1. base / step_31 / step_62 all run with the **current** behaviour, defects 1-3 fixed but the `args.get()` guard deliberately absent, so the three are mutually comparable.
2. Every exception enters the denominator and scores zero; `valid/total` and the exception rate are reported alongside accuracy.
3. After the three runs, re-run the same base over a fixed 60-task subset twice with the same seed and measure the task-level agreement. Until that number exists, no claim of reproducibility is made.
4. The `isinstance(args, dict)` guard lands in the **next** protocol version, documented as a change in environment feedback semantics rather than a bug fix folded into this curve.

## 10h. Judge switched to gpt-5.4-nano, with retry and alarm (2026-08-28)

The Qwen3.8-27B judge was taken offline by the user, so the reward model had to change mid-run. `gpt-5.4-nano` was chosen because finqa.md's own single-table judge was `gpt-5-nano`, making this the closest analogue to the paper's configuration.

Repository change is confined to `cookbooks/finqa/finqa_eval.py` (now +62/-5 lines; `verl` remains untouched, `git status` clean at `d8acd86e`). Both judge models are env-overridable so nothing is hard-coded to one deployment:

```python
JUDGE_MODEL = os.environ.get("FINQA_JUDGE_MODEL", "gpt-5.4-nano")
MULTI_TABLE_JUDGE_MODEL = os.environ.get("FINQA_MULTI_TABLE_JUDGE_MODEL", "gpt-5.4-mini")
```

Temperature is deliberately **not** set — the server default is used, per the user's instruction, and `max_output_tokens` stays at the production value of 512.

### Why a retry loop was required

In the Responses API there is no `finish_reason` field; the equivalent state is split across `response.status` (`completed` / `incomplete`) and `response.incomplete_details.reason` (`max_output_tokens`, `content_filter`, ...). The original code parsed `output_text` unconditionally, so a truncated or filtered response still produced a verdict. Since `_call_judge` decides correctness with `("true" in text) and ("false" not in text)`, a cut-off deliberation yields an essentially arbitrary reward.

Every response's termination state is now captured, never inferred. A response counts as usable only when `finish == "stop"` **and** the text is non-empty; otherwise it is retried up to `FINQA_JUDGE_MAX_ATTEMPTS` (10) with no wait between attempts, breaking on the first clean result. Ten consecutive abnormal terminations print an explicit alarm line and score the sample incorrect. Every attempt is appended to `logs/judge_finish.tsv` as `timestamp, finish, attempt, ok`.

Known gap: `judge_finish.tsv` is shared by the `.16` training run and the `.29` evaluations with **no source tag**, so the two streams cannot currently be separated. Adding a run identifier is outstanding.

### The truncation bias is one-directional

Measured on the Qwen3.8 judge before the switch: 6.3% of requests server-side finished with reason `length`. Under the old code that fraction became noise in the reward. The direction matters — truncation empties `output_text`, and an empty string makes `("true" in text)` deterministically False. So truncation could only ever destroy a positive reward, never manufacture one.

A controlled test (`judge_truncation.py`, 40 verbose-but-correct answers at cap 512 vs cap 4096) found 40/40 verdict agreement and no truncation at either cap, i.e. the failure needs answers longer than this synthetic set produces. That bounds the effect on ordinary cases; it does not repeal the mechanism, which is why the retry loop was added rather than declared unnecessary.

## 10i. Four-way judge A/B (2026-08-28)

Judges agree trivially on clearly-right and clearly-wrong answers, so those cases carry no information. `judge_ab.py` therefore builds controlled perturbations of real ground-truth answers and compares four judges on the surface that actually decides whether the reward is inflated or deflated: format tolerance and near misses.

```
| Judge          | Endpoint         | Unusable responses  | Termination note        |
|----------------|------------------|---------------------|-------------------------|
| Qwen3.8-27B    | .29:1702 (local) | 26/393 (6.6%)       | max_output_tokens only  |
| gpt-5.4-mini   | Azure            | 0/393 (0.0%)        | all completed           |
| gpt-5.6-luna   | Azure            | 0/393 (0.0%)        | all completed           |
| gpt-5.4-nano   | Azure            | 0/393 (0.0%)        | all completed           |
```

Two findings. First, **no Azure content filter was triggered** on any request across all three Azure tiers — the failure mode the user asked to check for does not occur on this rubric. Second, the only judge that failed to return usable verdicts was the self-hosted one, at 6.6% (26/393), which matches the 6.3% server-side `length` rate independently observed on the live endpoint.

On the cases where all four returned a verdict, unanimity was **98.9%**. The judges are effectively interchangeable as rulers on this task, which is what makes the mid-run switch defensible: the reward definition did not meaningfully move.

Cost per judge call is small and stable: 618 prompt tokens, 59 completion tokens. The short completions are a property of the rubric (the judge is asked for a terse verdict), not evidence of truncation.

## 10j. The evaluation could not resolve the model differences (2026-08-28)

Three models were evaluated on `.29` under the fixed protocol of section 10g (gpt-5.4-nano judge, `temperature=0.6, top_p=0.95, seed=1234`, one split per GPU):

```
| Model   | val correct/total | val score | val err | test correct/total | test score | test err |
|---------|-------------------|-----------|---------|--------------------|------------|----------|
| base    | 341/522           | 0.6533    | 1       | 366/558            | 0.6559     | 3        |
| step_31 | 340/522           | 0.6513    | 1       | 370/558            | 0.6631     | 2        |
| step_62 | 337/522           | 0.6456    | 0       | 368/558            | 0.6595     | 2        |
```

Then the reproducibility check of agreed-sequence item 3 was run — same base model, same seed, same 522 val rows, full scale rather than the planned 60-task subset:

```
| Run       | val correct/total | val score | val err |
|-----------|-------------------|-----------|---------|
| base      | 341/522           | 0.6533    | 1       |
| repro_A   | 337/522           | 0.6456    | 1       |
| repro_B   | 319/522           | 0.6111    | 4       |
```

**Spread: 4.21 pt on an identical model.** Every model-vs-base delta in the table above (-0.20 pt, -0.77 pt on val) is 5-20x smaller than the noise floor of the instrument that produced it.

### What this invalidates

The three-model table **cannot support any statement about training effect**, in either direction. It does not show that 62 steps helped, and it does not show that 62 steps did nothing. The earlier characterisation of the result as "no discernible difference, consistent with too few updates" was over-reading: the measurement had insufficient resolution to say anything, and "consistent with expectation" was an interpretation laid on top of an uninformative number.

The weight-delta measurements remain valid, because they were taken directly rather than through the evaluation: step_31 mean absolute delta 5.53e-09, step_62 1.358e-08, ratio 2.4 against a step ratio of 2. Relative to bf16 precision (3.9e-3) that is ~7e-7, i.e. only ~0.02% of elements moved by even one ULP. That is evidence about **weights**, not about **behaviour**, and the two must not be conflated.

### Why the seed did not help

`--concurrency 32` means vLLM batches requests continuously, and batch composition varies between runs. Different batch shapes change the floating-point reduction order in the matmuls, which perturbs logits in the last bits. At `temperature=0.6` that is sometimes enough to flip a sampled token, and one flipped token early in an agent trajectory forks the whole episode. A fixed RNG seed makes the *random draws* identical; it does nothing about the *distribution* those draws are applied to.

### Consequence: greedy is now the evaluation default

`eval_parallel.sh` defaults to `EVAL_TEMP=0, EVAL_TOP_P=1.0`. Argmax is far more robust to last-bit logit perturbation than sampling is, because a perturbation only matters when it reorders the top two candidates.

This makes standalone checkpoint evaluation a **different protocol** from in-training validation, deliberately:

```
| Protocol             | Sampling            | Question it answers               | Comparable to     |
|----------------------|---------------------|-----------------------------------|-------------------|
| In-training val      | temp 0.6 / p 0.95   | Expected performance of the policy| finqa.md Pass@1   |
| Standalone ckpt eval | temp 0 (greedy)     | Is model A different from model B?| other greedy runs |
```

Numbers from the two protocols are not interchangeable. Part of the gap between the in-training base line (0.6226) and the standalone base line (0.6533) is this protocol difference, not a model or judge difference — a distinction not drawn clearly enough when those two numbers were first reported side by side.

## 10k. Greedy decoding did not fix it either — the evaluation's resolution floor (2026-08-28)

Greedy was tried as the instrument fix. A criterion was fixed **before** the data was seen, so the verdict could not be chosen after the fact:

```
| |G1 - G2| in tasks, both splits | Verdict                   | Interpretable delta   |
|--------------------------------|---------------------------|-----------------------|
| 0                              | fully deterministic       | any delta >= 1 task   |
| 1-3                            | usable, floor 0.2-0.6 pt  | only deltas above 1 pt|
| > 3                            | greedy does not fix it    | needs repeats          |
```

Two greedy runs of the base model, same seed, same rows:

```
| Run          | val correct/total | val score | test correct/total | test score |
|--------------|-------------------|-----------|--------------------|------------|
| base (run 1) | 331/522           | 0.6341    | 371/558            | 0.6649     |
| base (run 2) | 337/522           | 0.6456    | 363/558            | 0.6505     |
| difference   | 6 tasks (1.15 pt) |           | 8 tasks (1.43 pt)  |            |
```

Verdict FAIL (6 and 8 against a gate of 3). The chain script stopped there and did **not** run the step_31 / step_62 evaluations, so no GPU time went into a measurement that could not have resolved them.

### The aggregate improvement was cancellation, not stability

Comparing aggregate scores made greedy look 3.7x better than sampling (1.15 pt vs 4.21 pt). Per-task pairing shows that reading was wrong:

```
| Pair              | Split | Protocol | n   | agree       | discordant  | net |
|-------------------|-------|----------|-----|-------------|-------------|-----|
| greedy G1 vs G2   | val   | greedy   | 522 | 452 (86.6%) | 70 (13.4%)  |  +6 |
| greedy G1 vs G2   | test  | greedy   | 558 | 496 (88.9%) | 62 (11.1%)  |  -8 |
| base vs repro_A   | val   | sampled  | 522 | 458 (87.7%) | 64 (12.3%)  |  -4 |
| base vs repro_B   | val   | sampled  | 522 | 430 (82.4%) | 92 (17.6%)  | -22 |
| repro_A vs repro_B| val   | sampled  | 522 | 432 (82.8%) | 90 (17.2%)  | -18 |
```

Greedy's per-task flip rate (13.4%) is **not** below the best sampled pair (12.3%). The favourable aggregate came from 32 up-flips nearly cancelling 38 down-flips. A single aggregate difference is itself a noise draw and cannot be used to estimate noise — an error made and corrected here.

### Resolution floor, derived rather than guessed

Under the null (same model), the variance of the net difference is the number of discordant pairs, so its sd is sqrt(70) = 8.4 tasks = **1.6 pt on 522 rows**. Observed +6 and -8 are 0.7 and 1.0 sd: exactly what an identical model should produce.

This also explains why pairing does not rescue the design. Pairing removes **task-difficulty** variance; the noise here is **per-task execution** randomness, which pairing leaves untouched. To resolve a 1 pt effect at 2 sd, roughly 10 repeats per model would be needed (~6.7 h of GPU each).

### The instability lives in long trajectories

```
| Group      | Tasks | Mean steps used |
|------------|-------|-----------------|
| Concordant | 452   | 4.84            |
| Discordant | 70    | 5.57            |
```

The 452-task stable core splits 299 correct / 153 wrong. The floating band is 70 tasks decided by run-to-run numerics. More steps means more opportunities for a last-bit logit perturbation to flip a token; in an agentic loop one changed SQL query changes the observation and forks everything downstream. Greedy removes sampling randomness but not the perturbation itself, which is why argmax barely helps.

### Root causes are removable, not merely averagable

```
| Setting               | During G1/G2 | Why it makes numerics history-dependent      |
|-----------------------|--------------|----------------------------------------------|
| enable_prefix_caching | True         | cache hit vs recompute are different paths;  |
|                       |              | whether it hits depends on concurrent history|
| enforce_eager         | False        | cudagraphs captured per batch-shape bucket   |
| concurrency           | 32           | batch composition varies -> reduction order  |
```

The first two are removable at ~20-30% throughput. The third is only removable at `--concurrency 1`, i.e. 32x wall clock. `det_D1` / `det_D2` test the first two (`VLLM_EXTRA="--no-enable-prefix-caching --enforce-eager"`, both confirmed in force in the vLLM log).

### The better instrument was already running

The in-training reward curve does not go through this channel at all: 256 rollouts per step, 62 steps, 15,872 rollouts, and it is produced by the training loop itself. It resolves ~2 pt (section 10j) and it is free. Standalone evaluation should be reserved for the end of the run, with repeats, once there is an effect large enough to clear a 1.6 pt floor.

## 10l. Root cause found: batch composition, proved directly (2026-08-28)

Four instrument fixes had failed in a row, so instead of a fifth guess the mechanism was tested directly, at the prompt level, with no agent loop involved (`batch_determinism.py`). The same prompt, `temperature=0`, `seed=1234`, sent two ways:

```
| Regime                            | Distinct outputs | Verdict          |
|-----------------------------------|------------------|------------------|
| Serial, one request at a time     | 1/8              | DETERMINISTIC    |
| Same prompt in a 32-wide burst    | 5/8              | NONDETERMINISTIC |
```

Serial decoding is bit-reproducible. Put the identical request inside a concurrent burst and it yields five different completions out of eight. One of the batched outputs equals the serial one, so this is not "batching takes a different code path" — batch composition jitters the result among several numerically valid outcomes.

### Why the earlier attempts missed it

```
| Attempt                       | Targeted                     | val flip rate |
|-------------------------------|------------------------------|---------------|
| Pinned seed, temp 0.6 / 0.95  | random draws                 | 12.3%         |
| Greedy, temp 0                | sampling randomness          | 13.4%         |
| + --no-enable-prefix-caching  | cache-hit vs recompute path  | 14.6%         |
| + --enforce-eager             | cudagraph shape buckets      |               |
```

None of these touch batch composition, which is why the flip rate never moved. Only `--concurrency 1` removes it, at roughly 32x wall clock.

### Amplification is by sequence length, not by a high per-token rate

With prompt provably identical (system and user messages byte-equal) and tool-call IDs stripped, 48% of step-0 responses still differ. For a ~250-token response that implies a per-token divergence probability around 1/385. A rare event, amplified twice: once by response length, then again by the agent loop, where one changed SQL query changes the observation and forks everything downstream. Whole-trajectory identity is 0/522.

### Three measurement errors made along the way, all corrected

```
| Claim made                          | Why it was wrong                        |
|-------------------------------------|-----------------------------------------|
| "greedy cut noise 3.7x"             | used a single aggregate difference as a |
|                                     | noise estimate; that number IS a noise  |
|                                     | draw. Per-task flip rate was unchanged. |
| "the step-0 prompt already differs" | compared chat_completions including the |
|                                     | assistant reply; system+user are equal. |
| "the judge is the noise source"     | refuted by the data: 70 of 70 verdict   |
|                                     | flips had a different agent answer.     |
```

The tool-call `id` field (`chatcmpl-tool-<random hex>`) is regenerated per request and must be stripped before any trajectory comparison, or everything looks different.

### Consequences

This is a permanent property of any concurrent vLLM evaluation here, not a bug in this run. It does **not** affect training: GRPO rollouts are meant to be stochastic (temperature 0.7), and this jitter is simply part of that distribution.

For evaluation it fixes the resolution floor at about **1.6 pt (1 sd) on 522 rows**, so:

1. Do not evaluate checkpoints whose expected effect is below ~3 pt; the in-training reward curve (256 rollouts/step, ~2 pt resolution over 20-step windows) is both cheaper and sharper.
2. When a standalone number is finally needed, budget repeats: k runs cut the sd by sqrt(k), so 4 repeats reach 0.8 pt and 10 reach 0.5 pt.
3. Report every standalone score with the run count and the floor, never as a bare accuracy.

## 11. Open questions

1. ~~Checkpoint writing unverified~~ — **resolved**, verified at step 10 (section 9.4).
2. ~~Validation loop unverified~~ — **resolved**, `batch/time/evaluator_s` 26.3 at step 10.
3. **~46 h for one epoch.** Options, quantified: raising `ppo_max_token_len_per_gpu` 12288 -> 16384 (~15%, no deviation from finqa.md, but logits would grow to ~41 GiB of 47.36 and we already hit one `wake_up` OOM); dropping KL (~7 h, deviates); halving the training set (~23 h, no longer a full epoch).
4. **No local base line yet.** `val_before_train=true` will produce one over the full 522-task val split at step 0 of the epoch run. A full `test`/`multi_test` base line still needs `eval_full.sh base`.
5. **`right_table_access_reward` is computed but not added to the reward** — this is deliberate, matching finqa.md Finding 2 (partial rewards scored 54.0% vs binary 66.3%). Do not "fix" it.

---

## 12. Post-training evaluation plan

`./eval_full.sh {base|/path/to/merged_hf} [tag]` serves the model with vLLM and evaluates all three sets:

```
| Eval set          | Tasks | Type         | Purpose                              |
|-------------------|-------|--------------|--------------------------------------|
| finqa/val         |   522 | single-table | same set the training loop validates |
| finqa/test        |   558 | single-table | primary final metric                 |
| finqa/multi_test  |   131 | multi-table  | held-out generalization, never trained|
```

Checkpoint merge: `python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor --target_dir <hf_dir>`.

Both base and trained models must be measured under identical conditions; finqa.md's numbers are not directly comparable because the benchmark and the judge both differ.

---

## 13. Run history

```
| Time  | Host | Run                          | Outcome                                  |
|-------|------|------------------------------|------------------------------------------|
| 12:18 | .29  | smoke (LoRA broken)          | OOM in optimizer_step, 28.86 GiB         |
| 12:34 | .29  | smoke (seq 5120)             | identical OOM -> led to the LoRA bug     |
| 12:44 | .29  | smoke (LoRA fixed)           | 2 steps OK, peak 13.55 GiB               |
| 13:53 | .16  | vllm_probe                   | 137*24 -> 3288, mixed cu12/cu13 verified |
| 13:53 | .16  | smoke (batch 10, no KL)      | 2 steps OK, peak 24.98 GiB, 1410 tok/s   |
| 14:31 | .16  | literal (2048/16384)         | 188 rejections (14.1%), 0 steps in 15 min|
| 14:35 | .16  | stability (offload off)      | wake_up OOM at step 1                    |
| 15:02 | .16  | stability (offload on)       | 20/20 steps, all 7 checks PASSED          |
| 09:52 | .16  | epoch (total_epochs=1)       | base 0.6054; superseded (test_freq wrong) |
| 10:2x | .16  | epoch (total_epochs=10)      | base 0.6226; later restarted              |
| 08-28 | .29  | eval base / step_31 / step_62| 3 models x 2 splits, sampled protocol     |
| 08-28 | .29  | repro_A / repro_B (base)     | 341 / 337 / 319 -> protocol has 4.21 pt   |
|       |      |                              | noise; three-model table unusable         |
| 12:27 | .16  | epoch resumed from step_62   | RUNNING pid 2326140, at step 68           |
| 08-28 | .29  | greedy_G1 / greedy_G2 (base) | greedy reproducibility test, in progress  |
```
