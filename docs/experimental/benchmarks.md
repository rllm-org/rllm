# Supported Benchmarks

rLLM ships with **57 benchmarks** across 10 categories, all accessible via `rllm eval <benchmark>`. Datasets are auto-pulled from HuggingFace on first use and cached locally under `~/.rllm/datasets/`.

## Quick Reference

```bash
rllm eval gsm8k --max-examples 10          # Run with defaults from catalog
rllm eval mmlu_pro --model gpt-4o          # Override model
rllm eval humaneval --agent code --split test  # Explicit agent and split
rllm dataset list                           # List all available benchmarks
```

---

## Math (8 benchmarks)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| GSM8K | `gsm8k` | 8.5K train, 1.3K test | Grade school math word problems | `math` | `math_reward_fn` |
| MATH-500 | `math500` | 500 test | Competition-level math from the MATH benchmark | `math` | `math_reward_fn` |
| Countdown | `countdown` | 1K train, 500 test | Arithmetic puzzle — reach a target using given numbers | `countdown` | `countdown_reward_fn` |
| HMMT Feb 2025 | `hmmt` | train | Harvard-MIT Mathematics Tournament (February) | `math` | `math_reward_fn` |
| HMMT Nov 2025 | `hmmt_nov` | 30 train | Harvard-MIT Mathematics Tournament (November) | `math` | `math_reward_fn` |
| PolyMATH | `polymath` | 4 splits (top/high/medium/low) | Multilingual math reasoning across 18 languages | `math` | `math_reward_fn` |
| AIME 2025 | `aime_2025` | 30 train | American Invitational Mathematics Examination 2025 | `math` | `math_reward_fn` |
| AIME 2026 | `aime_2026` | 30 train | American Invitational Mathematics Examination 2026 | `math` | `math_reward_fn` |

**Agent:** `math` prompts the model for step-by-step reasoning and a final `\boxed{}` answer. `countdown` uses `<answer>` tags.

**Evaluator:** `math_reward_fn` extracts boxed answers and compares using symbolic math equivalence (via sympy). `countdown_reward_fn` validates the arithmetic equation against the target.

---

## Multiple Choice (10 benchmarks)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| MMLU-Pro | `mmlu_pro` | 12K test | Expert-level MCQ with 10 options | `mcq` | `mcq_reward_fn` |
| MMLU-Redux | `mmlu_redux` | 3K test | Curated MMLU subset with error corrections | `mcq` | `mcq_reward_fn` |
| GPQA | `gpqa_diamond` | 448 train | Graduate-level science QA (physics, chemistry, biology) | `mcq` | `mcq_reward_fn` |
| SuperGPQA | `supergpqa` | 26.5K test | Large-scale graduate QA across 285 disciplines | `mcq` | `mcq_reward_fn` |
| C-Eval | `ceval` | 13.9K val | Chinese evaluation benchmark across 52 disciplines | `mcq` | `mcq_reward_fn` |
| MMMLU | `mmmlu` | 15.9K/lang test | Multilingual MMLU across 14 languages | `mcq` | `mcq_reward_fn` |
| MMLU-ProX | `mmlu_prox` | 11.8K/lang test | Multilingual MMLU-Pro across 29 languages | `mcq` | `mcq_reward_fn` |
| INCLUDE | `include` | test | Multilingual knowledge from local exams, 44 languages | `mcq` | `mcq_reward_fn` |
| Global PIQA | `global_piqa` | test | Physical commonsense reasoning, 100+ languages | `mcq` | `mcq_reward_fn` |
| LongBench v2 | `longbench_v2` | test | Long-context understanding MCQ | `mcq` | `mcq_reward_fn` |

**Agent:** `mcq` formats options as A–J, prompts for a single letter answer.

**Evaluator:** `mcq_reward_fn` extracts the letter choice and does exact match against ground truth.

**Note:** `mmlu_redux`, `ceval`, `mmlu_prox`, `include`, and `global_piqa` use `aggregate_configs` to merge all HuggingFace sub-configs (one per subject/language) into a single dataset with a `language` column.

---

## Code (3 benchmarks)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| HumanEval | `humaneval` | 164 test | Function-level code generation | `code` | `code_reward_fn` |
| MBPP | `mbpp` | 974 test | Python programming problems | `code` | `code_reward_fn` |
| LiveCodeBench | `livecodebench` | test | Contamination-free competitive programming | `code` | `code_reward_fn` |

**Agent:** `code` prompts the model to generate a Python solution wrapped in ` ```python ``` ` markers.

**Evaluator:** `code_reward_fn` executes the generated code against test cases in a sandbox. Reward = fraction of tests passed.

---

## Question Answering (3 benchmarks)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| HotpotQA | `hotpotqa` | 7.4K validation | Multi-hop question answering with distractor paragraphs | `qa` | `f1_reward_fn` |
| AA-LCR | `aa_lcr` | 100 test | Long-context reasoning over ~100K-token documents | `qa` | `llm_equality_reward_fn` |
| HLE | `hle` | 2,500 test | Humanity's Last Exam — expert-level across dozens of subjects | `reasoning` | `llm_equality_reward_fn` |

**Agents:** `qa` prompts for concise answers. `reasoning` uses chain-of-thought with an `ANSWER:` marker.

**Evaluators:**
- `f1_reward_fn` — token-overlap F1 score between prediction and ground truth.
- `llm_equality_reward_fn` — pipeline: exact normalized match → LLM semantic judge → F1 fallback.

> **HLE is gated.** Requires `HF_TOKEN` environment variable or `huggingface-cli login` to access.

---

## Instruction Following (2 benchmarks)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| IFEval | `ifeval` | 541 train | Instruction following with verifiable constraints | `ifeval` | `ifeval_reward_fn` |
| IFBench | `ifbench` | test | Out-of-distribution instruction following | `ifeval` | `ifeval_reward_fn` |

**Agent:** `ifeval` passes the prompt directly to the model — the instructions themselves contain the constraints.

**Evaluator:** `ifeval_reward_fn` checks 22+ constraint types: keyword existence/frequency, word/sentence/paragraph count, formatting (bullets, JSON, title case), case rules, language requirements, and more.

---

## Agentic / Tool Use (2 benchmarks)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| BFCL | `bfcl` | test | Berkeley Function Calling Leaderboard (exec_simple) | `bfcl` | `bfcl_reward_fn` |
| MultiChallenge | `multichallenge` | test | Multi-turn conversation evaluation | `multiturn` | `llm_judge_reward_fn` |

**Agents:**
- `bfcl` injects tool/function definitions and expects the model to make tool calls with correct arguments.
- `multiturn` replays a multi-turn conversation, then poses a final target question.

**Evaluators:**
- `bfcl_reward_fn` — AST-level comparison of tool calls (function name + arguments) against ground truth.
- `llm_judge_reward_fn` — LLM-as-judge scores the response against a rubric/pass-criteria.

---

## Search (4 benchmarks)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| BrowseComp | `browsecomp` | 830 test | Web browsing comprehension (encrypted, auto-decrypted) | `search` | `llm_equality_reward_fn` |
| Seal-0 | `seal0` | 111 test | Search-augmented QA with freshness metadata | `search` | `llm_equality_reward_fn` |
| WideSearch | `widesearch` | 200 test | Broad web search with structured table output | `search` | `widesearch_reward_fn` |
| HLE + Search | `hle_search` | 2,500 test | Humanity's Last Exam with web search tools | `search` | `llm_equality_reward_fn` |

**Agent:** `search` is a multi-turn tool-calling agent that uses web search (Serper or Brave) to find information and answers in `\boxed{}` format.

**Evaluators:**
- `llm_equality_reward_fn` — exact match → LLM semantic judge → F1 fallback (used for most search benchmarks).
- `widesearch_reward_fn` — parses markdown table output and computes row-level F1 against gold structured spec.

**Setup:** Requires a search API key. Set `SERPER_API_KEY` (recommended, $0.30/1K queries) or `BRAVE_API_KEY`. Override with `--search-backend serper|brave`.

> **HLE + Search is gated.** Requires `HF_TOKEN` environment variable or `huggingface-cli login` to access.

---

## Translation (1 benchmark)

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| WMT24++ | `wmt24pp` | train | Machine translation across 55 language pairs | `translation` | `translation_reward_fn` |

**Agent:** `translation` prompts the model to translate text to a target language, outputting only the translation.

**Evaluator:** `translation_reward_fn` computes ChrF (character n-gram F-score), a language-agnostic MT metric.

---

## Vision-Language (23 benchmarks)

### Visual Math & Understanding

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| MMMU | `mmmu` | 900 validation | Multi-discipline multimodal understanding | `vlm_mcq` | `mcq_reward_fn` |
| MMMU-Pro | `mmmu_pro` | 1,730 test | Harder MMMU with 10 options | `vlm_mcq` | `mcq_reward_fn` |
| MathVision | `mathvision` | 304 testmini | Visual math reasoning | `vlm_math` | `math_reward_fn` |
| MathVista | `mathvista` | 1,000 testmini | Visual math across diverse task types | `vlm_math` | `math_reward_fn` |
| DynaMath | `dynamath` | 501 test | Dynamic visual math with generated variants | `vlm_math` | `math_reward_fn` |
| ZEROBench | `zerobench` | 100 test | Zero-shot visual reasoning | `vlm_open` | `llm_equality_reward_fn` |
| ZEROBench Sub | `zerobench_sub` | 334 test | Decomposed visual reasoning subquestions | `vlm_open` | `llm_equality_reward_fn` |
| VLMs Are Blind | `vlmsareblind` | 8,020 test | Visual perception (counting, spatial, etc.) | `vlm_open` | `f1_reward_fn` |
| BabyVision | `babyvision` | 388 test | Early visual understanding (MCQ + fill-blank) | `vlm_open` | `llm_equality_reward_fn` |
| Geometry3K | `geo3k` | 2.4K train, 601 test | Geometry problem solving with diagrams | `vlm_math` | `math_reward_fn` |

### Text Recognition & Document Understanding

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| AI2D | `ai2d` | 3,088 test | Science diagram understanding MCQ | `vlm_mcq` | `mcq_reward_fn` |
| OCRBench | `ocrbench` | 1,000 test | OCR and text recognition | `vlm_open` | `f1_reward_fn` |
| CharXiv | `charxiv` | 1,000 validation | Chart understanding reasoning questions | `vlm_open` | `llm_equality_reward_fn` |
| CC-OCR | `cc_ocr` | 7,058 test | Multi-scene OCR with 4 sub-tasks (base64 images) | `vlm_open` | `f1_reward_fn` |
| OmniDocBench | `omnidocbench` | test | Comprehensive document understanding | `vlm_open` | `f1_reward_fn` |
| DocVQA | `docvqa` | 5,188 validation | Single-page document visual QA | `vlm_open` | `f1_reward_fn` |

### Spatial Intelligence

| Benchmark | CLI Name | Size | Description | Agent | Evaluator |
|---|---|---|---|---|---|
| CountBenchQA | `countbenchqa` | 491 test | Visual object counting QA | `vlm_open` | `f1_reward_fn` |
| ERQA | `erqa` | 400 test | Entity recognition QA (multi-image) | `vlm_mcq` | `mcq_reward_fn` |
| RefCOCO | `refcoco` | test | Referring expression comprehension (bbox grounding) | `vlm_grounding` | `iou_reward_fn` |
| RefSpatial-Bench | `refspatial` | test | Referential spatial reasoning (point prediction) | `vlm_open` | `point_in_mask_reward_fn` |
| LingoQA | `lingoqa` | test | Language-grounded QA for autonomous driving | `vlm_open` | `f1_reward_fn` |
| SUN RGB-D | `sunrgbd` | test | Depth estimation and scene understanding | `vlm_open` | `depth_reward_fn` |

**Agents:**
- `vlm_mcq` — formats image + options, expects single letter answer.
- `vlm_math` — image-based math reasoning, expects `\boxed{}` answer.
- `vlm_open` — open-ended visual QA, free-form response.
- `vlm_grounding` — visual grounding, expects `[x1, y1, x2, y2]` bounding box output.

All VLM agents embed images as base64 in OpenAI-compatible multimodal content blocks, supporting PNG, JPEG, and WebP formats.

**Note:** `cc_ocr` uses `aggregate_configs` to merge 4 sub-tasks (doc_parsing, kie, multi_lan_ocr, multi_scene_ocr) and its images are base64-encoded strings in HuggingFace (decoded to bytes by the transform). `erqa` supports multi-image inputs (1-16 images per question). `mmlongbench_doc` involves multi-page PDFs which may hit API limits for 40+ page documents. `lingoqa` is originally a video QA benchmark — the transform uses key frames as static images.

---

## Agents Summary

15 built-in agents are available, each implementing the `AgentFlow` protocol:

| Agent | Description |
|---|---|
| `math` | Step-by-step math reasoning → `\boxed{}` answer |
| `countdown` | Arithmetic puzzle solving → `<answer>` tag |
| `code` | Python code generation → ` ```python ``` ` block |
| `qa` | Concise question answering |
| `mcq` | Multiple choice selection → single letter (A–J) |
| `ifeval` | Instruction following (pass-through) |
| `bfcl` | Function/tool calling with definitions |
| `multiturn` | Multi-turn conversation replay |
| `reasoning` | Chain-of-thought → `ANSWER:` marker |
| `translation` | Text translation → target language output |
| `search` | Multi-turn web search tool calling → `\boxed{}` answer |
| `vlm_mcq` | Multimodal MCQ with images |
| `vlm_math` | Multimodal math with images |
| `vlm_open` | Multimodal open-ended QA |
| `vlm_grounding` | Visual grounding → `[x1, y1, x2, y2]` bounding box |

---

## Evaluators Summary

14 built-in evaluators score agent outputs:

| Evaluator | Method |
|---|---|
| `math_reward_fn` | Symbolic math equivalence (sympy) |
| `countdown_reward_fn` | Equation validation against target |
| `code_reward_fn` | Sandboxed test case execution |
| `f1_reward_fn` | Token-overlap F1 score |
| `mcq_reward_fn` | Exact letter match |
| `ifeval_reward_fn` | 22+ constraint type verification |
| `bfcl_reward_fn` | Function call AST matching |
| `llm_judge_reward_fn` | LLM-as-judge with rubric |
| `llm_equality_reward_fn` | Exact match → LLM judge → F1 fallback |
| `translation_reward_fn` | ChrF character n-gram score |
| `widesearch_reward_fn` | Structured table row-level F1 matching |
| `iou_reward_fn` | Bounding box IoU ≥ 0.5 for visual grounding |
| `point_in_mask_reward_fn` | Point-in-mask check for spatial reasoning |
| `depth_reward_fn` | Absolute relative error for depth estimation |

---

## Adding a New Benchmark

1. **Add a catalog entry** to `rllm/registry/datasets.json`:
   ```json
   "my_benchmark": {
     "description": "Short description (N examples)",
     "source": "huggingface/repo-name",
     "category": "math|mcq|code|qa|...",
     "splits": ["test"],
     "default_agent": "math",
     "reward_fn": "math_reward_fn",
     "eval_split": "test",
     "transform": "rllm.data.transforms:my_transform"
   }
   ```

2. **Add a transform** (if needed) to `rllm/data/transforms.py` to normalize HuggingFace fields to the expected format (`question`, `ground_truth`, `options`, etc.).

3. **Optional catalog fields:**
   - `hf_config` — specific HuggingFace dataset configuration name
   - `hf_split` — override the HF split name (e.g., load `"train"` but register as `"test"`)
   - `aggregate_configs` — merge all HF configs into one dataset with a `language` column
   - `field_map` — rename fields (e.g., `{"prompt": "question"}`)
   - `data_files` — specific data file within the HF repo
   - `gated` — mark as requiring HuggingFace authentication

4. **Run it:**
   ```bash
   rllm eval my_benchmark --max-examples 5
   ```

See [eval-framework.md](eval-framework.md) for full details on the eval pipeline, custom agents, and custom evaluators.
