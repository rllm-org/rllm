"""Path constants for the finqa cookbook.

Module name is prefixed with ``finqa_`` to avoid colliding with other
cookbooks that ship a top-level ``constants.py`` after editable install.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
PROMPTS_DIR = PROJECT_DIR / "prompts"

# The company-tables tree is large (~6.9k tables); allow override.
TABLES_ROOT = Path(os.getenv("FINQA_TABLES_ROOT", str(DATA_DIR / "company_tables")))
TABLES_CLEANED_ALL_COMPANIES_FILE_NAME = "tables_cleaned_all_companies.json"

# Question-set CSV paths
TRAIN_QUESTIONS_PATH = DATA_DIR / "train_finqa.csv"
VAL_QUESTIONS_PATH = DATA_DIR / "val_finqa.csv"
TEST_QUESTIONS_PATH = DATA_DIR / "test_finqa.csv"

# Prompt files
REACT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "react_system_prompt.txt"
CORRECTNESS_PROMPT_PATH = PROMPTS_DIR / "correctness_prompt.txt"
MULTI_TABLE_CORRECTNESS_PROMPT_PATH = PROMPTS_DIR / "multi_table_correctness_prompt.txt"
