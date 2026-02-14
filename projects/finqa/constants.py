import os
from pathlib import Path

# Base directories
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
PROMPTS_DIR = PROJECT_DIR / "prompts"
MULTI_TABLE_DATA_DIR = DATA_DIR / "multi_table_data"

# Tables
TABLES_ROOT = Path(os.getenv("FINQA_TABLES_ROOT", str(DATA_DIR / "company_tables")))
TABLES_CLEANED_ALL_COMPANIES_FILE_NAME = "tables_cleaned_all_companies.json"

# Dataset paths
COMPANY_SPLIT_PATH = DATA_DIR / "split.json"
TRAIN_QUESTIONS_PATH = DATA_DIR / "train_finqa.csv"
VAL_QUESTIONS_PATH = DATA_DIR / "val_finqa.csv"
TEST_QUESTIONS_PATH = DATA_DIR / "test_finqa.csv"
MULTI_TABLE_TRAIN_PATH = MULTI_TABLE_DATA_DIR / "train_finqa.csv"
MULTI_TABLE_VAL_PATH = MULTI_TABLE_DATA_DIR / "val_finqa.csv"
MULTI_TABLE_TEST_PATH = MULTI_TABLE_DATA_DIR / "test_finqa.csv"

# Prompt paths
REACT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "react_system_prompt.txt"
CORRECTNESS_PROMPT_PATH = PROMPTS_DIR / "correctness_prompt.txt"
MULTI_TABLE_CORRECTNESS_PROMPT_PATH = PROMPTS_DIR / "multi_table_correctness_prompt.txt"

# Data generation paths (only needed by scripts/data_generation/)
SCRAPED_DATA_DIR = DATA_DIR / "scraped"
SCRAPED_DATA_URLS_PATH = SCRAPED_DATA_DIR / "seed_10k_urls.csv"
QUESTION_GENERATION_PROMPT_PATH = PROMPTS_DIR / "question_generation_prompt.txt"
QUESTION_VERIFICATION_PROMPT_PATH = PROMPTS_DIR / "question_verification_prompt.txt"
