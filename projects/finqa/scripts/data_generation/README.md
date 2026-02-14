# Data Generation

Three step pipeline for building Q&A data from SEC 10-K filings.

**Prerequisites:** Follow [FinQA installation](../../README.md#installation).

**Requirements:** Google Chrome (for Step 1), vLLM server running `Qwen/Qwen3-30B-A3B-Instruct-2507` (for Steps 2 and 3)

All commands below assume you are in the **repo root** (`rllm/`).

## Step 1: Download 10-K filings

Scrapes SEC EDGAR pages via headless Chrome, extracts all `TableTextBlock` HTML elements, saves each as a `.txt` file.

```bash
python projects/finqa/scripts/data_generation/download_10k.py \
  --user_agent "your-email@example.com" # required (SEC fair access policy)
```


**Output:** `<output_base_dir>/<company>/*.txt` — one raw HTML file per table.

## Step 2: Clean and structure tables

Sends each `.txt` to Qwen3-30B (vLLM at `localhost:32000`) to extract structured table data. Writes per-table `.json` files and a per-company summary.

```bash
python projects/finqa/scripts/data_generation/cleanup_tables.py \
  --input_base_dir projects/finqa/data/scraped/2025-01-28
```

**Output:** `*.json` files alongside each `.txt`, plus `tables_cleaned_all_companies.json` per company.

## Step 3: Generate questions

Generates and double-verifies Q&A pairs using Qwen3-30B (vLLM at `localhost:30000`). Only keeps questions whose column/row references exist in the source table.

Before running, copy cleaned company directories into the tables root and create a split file:

```bash
cp -r projects/finqa/data/scraped/<date>/* projects/finqa/data/company_tables/
# Create projects/finqa/data/split.json: {"train": ["company_a", ...], "val": [...], "test": [...]}
```

Then generate:

```bash
python projects/finqa/scripts/data_generation/generate_questions.py        # all splits
python projects/finqa/scripts/data_generation/generate_questions.py train   # single split
```

Reads table data from `projects/finqa/data/company_tables/` (or `FINQA_TABLES_ROOT` env var), prompts from `projects/finqa/prompts/`.

**Output:** `projects/finqa/data/train_finqa.csv`, `val_finqa.csv`, `test_finqa.csv` — each with question, answer, explanation, company, table name, and column/row references.
