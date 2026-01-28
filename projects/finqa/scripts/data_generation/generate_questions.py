# Standard library imports
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import pandas as pd
from openai import OpenAI

# Relative imports
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(BASE_DIR)

from src.constants import (
    COMPANY_SPLIT_PATH,
    FIN_QA_QUESTION_GENERATION_SYSTEM_PROMPT_PATH,
    FIN_QA_QUESTION_VERIFICATION_SYSTEM_PROMPT_PATH,
    TABLES_CLEANED_ALL_COMPANIES_FILE_NAME,
    TABLES_ROOT,
    TEST_QUESTIONS_PATH,
    TRAIN_QUESTIONS_PATH,
    VAL_QUESTIONS_PATH,
)

# Pattern to strip leading/trailing LaTeX boxed answers
BOXED_PATTERN = re.compile(r"\\boxed\{(?P<value>.+)\}$")

# Column order used for every split CSV
OUTPUT_COLUMNS = [
    "user_query",
    "id",
    "question_type",
    "question",
    "answer",
    "explanation",
    "company",
    "table_name",
    "columns_used_json",
    "rows_used_json",
]

# Destination CSVs for each split
OUTPUT_PATHS = {
    "train": Path(TRAIN_QUESTIONS_PATH),
    "val": Path(VAL_QUESTIONS_PATH),
    "test": Path(TEST_QUESTIONS_PATH),
}

VALID_SPLITS = ("train", "val", "test")
NUM_VERIFICATION_RUNS = 2

# Thread pool size
TABLE_WORKERS = 32
COMPANY_WORKERS = 2

GUIDED_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "question_type": {"type": "string"},
                "question": {"type": "string"},
                "explanation": {"type": "string"},
                "answer": {"type": "string"},
                "columns_used": {"type": "array", "items": {"type": "string"}},
                "rows_used": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "question_type",
                "question",
                "explanation",
                "answer",
                "columns_used",
                "rows_used",
            ],
            "additionalProperties": False,
        },
        {"type": "null"},
    ]
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"pass": {"type": "boolean"}},
    "required": ["pass"],
    "additionalProperties": False,
}


def strip_boxed(answer: str) -> str:
    """Remove a single outer \\boxed{} wrapper, if the model produced one"""
    value = str(answer).strip()
    match = BOXED_PATTERN.fullmatch(value)
    return match.group("value").strip() if match else value


def _normalize_token(text: str) -> str:
    """Convert text to a lowercase alphanumeric token for matching"""
    text = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_")


def normalize_table_references(columns: List[str], rows: List[str], table: pd.DataFrame):
    """Align loosely formatted column and row references with table values"""
    column_candidates = [str(value).strip() for value in columns if str(value).strip()]
    row_candidates = [str(value).strip() for value in rows if str(value).strip()]

    column_lookup = {_normalize_token(col): str(col) for col in table.columns}
    cell_values = table.fillna("").astype(str).values.flatten()
    row_lookup = {_normalize_token(value): value.strip() for value in cell_values if value and value.strip()}

    normalized_columns = [column_lookup.get(_normalize_token(col), col) for col in column_candidates]
    normalized_rows = [row_lookup.get(_normalize_token(row), row) for row in row_candidates]
    return normalized_columns, normalized_rows


def clean_question(question: Any, table: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Validate required fields from the generator and tidy each value"""
    if not isinstance(question, dict):
        return None
    try:
        cleaned = {key: str(question[key]).strip() for key in ("question_type", "question", "explanation", "answer")}
        columns_used = [value for value in (str(item).strip() for item in question["columns_used"]) if value]
        rows_used = [value for value in (str(item).strip() for item in question["rows_used"]) if value]
    except (KeyError, TypeError):
        return None

    columns_used, rows_used = normalize_table_references(columns_used, rows_used, table)

    if not validate_references(columns_used, rows_used, table):
        return None

    return {**cleaned, "columns_used": columns_used, "rows_used": rows_used}


def validate_references(columns_used: List[str], rows_used: List[str], table: pd.DataFrame) -> bool:
    """Confirm the referenced columns and row values exist in the table"""
    if any(col not in table.columns for col in columns_used):
        return False
    table_values = {cell.strip().lower() for cell in table.fillna("").astype(str).values.flatten() if cell and cell.strip()}
    return all(str(row).strip().lower() in table_values for row in rows_used)


def generate_question(client: OpenAI, system_prompt: str, payload: str) -> Optional[Dict[str, Any]]:
    """Call the generator model with the guided schema and parse the response"""
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        max_tokens=1024,
        temperature=0.5,
        top_p=0.95,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
        extra_body={"guided_json": GUIDED_SCHEMA},
    )
    content = completion.choices[0].message.content
    try:
        return json.loads(content) if content else None
    except json.JSONDecodeError:
        return None


def verify_question(client: OpenAI, verification_prompt: str, table: Dict, question: Dict, table_name: str) -> bool:
    """Ask the verifier model to confirm the question/answer fits the table"""
    user_message = (
        f"Table Name: {table_name}\n\n"
        f"Original Data:\n{json.dumps(table, indent=2)}\n\n"
        f"Question:\n{json.dumps(question, indent=2)}\n"
    )
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        max_tokens=50,
        temperature=0.0,
        top_p=0.3,
        messages=[
            {"role": "system", "content": verification_prompt},
            {"role": "user", "content": user_message},
        ],
        extra_body={"guided_json": VERIFY_SCHEMA},
    )
    content = completion.choices[0].message.content
    try:
        return bool(json.loads(content or "{}").get("pass"))
    except json.JSONDecodeError:
        return False


def process_table(client: OpenAI, system_prompt: str, verification_prompt: str, company: str, split_name: str, table_path: Path) -> Optional[Dict]:
    """Generate and verify one question for a company table"""
    table_name = table_path.stem
    table_json = json.loads(table_path.read_text())
    df = pd.DataFrame(table_json)
    columns_json = json.dumps(df.columns.tolist())
    records_json = json.dumps(df.reset_index(drop=True).to_dict(orient="records"))

    payload = (
        f"Company: {company}\n"
        f"Split: {split_name}\n"
        f"Table Name: {table_name}\n"
        f"Columns: {columns_json}\n"
        f"Rows:\n{records_json}\n"
        f"Original Table JSON:\n{json.dumps(table_json)}"
    )
    raw_question = generate_question(client, system_prompt, payload)
    if raw_question is None:
        return None

    question = clean_question(raw_question, df)
    if not question:
        return None

    if not all(
        verify_question(client, verification_prompt, table_json, question, table_name)
        for _ in range(NUM_VERIFICATION_RUNS)
    ):
        return None

    return {
        "user_query": f"For company `{company}`, here is the question: {question['question']}",
        **{key: question[key] for key in ("question_type", "question", "explanation")},
        "answer": strip_boxed(question["answer"]),
        "company": company,
        "table_name": table_name,
        "columns_used_json": json.dumps(question["columns_used"]),
        "rows_used_json": json.dumps(question["rows_used"]),
    }


def process_company(client: OpenAI, system_prompt: str, verification_prompt: str, company: str, split_name: str) -> List[Dict]:
    """Iterate through a company's tables and collect verified question rows"""
    company_dir = Path(TABLES_ROOT) / company
    if not company_dir.is_dir():
        print(f"{split_name}/{company}: missing directory")
        return []

    table_paths = [path for path in sorted(company_dir.glob("*.json")) if path.name != TABLES_CLEANED_ALL_COMPANIES_FILE_NAME]
    if not table_paths:
        print(f"{split_name}/{company}: no table files found")
        return []

    print(f"{split_name}/{company}: processing {len(table_paths)} tables...", flush=True)

    def run(path):
        try:
            return process_table(client, system_prompt, verification_prompt, company, split_name, path)
        except Exception as exc:
            print(f"{split_name}/{company}/{path.stem}: error {exc}")
            return None

    with ThreadPoolExecutor(max_workers=TABLE_WORKERS) as executor:
        rows = [row for row in executor.map(run, table_paths) if row]
    print(f"{split_name}/{company}: {len(rows)} questions")
    return rows


def finalize_split(split_name: str, rows: List[Dict], output_path: Path) -> None:
    """Write the split CSV with deterministic IDs when we have any rows"""
    if not rows:
        print(f"{split_name}: no questions")
        return

    df = pd.DataFrame(rows).sort_values(["company", "table_name", "question"]).reset_index(drop=True)
    df["id"] = df.index
    df = df[OUTPUT_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"{split_name}: wrote {len(df)} rows to {output_path}")


def main(target_split: Optional[str] = None) -> None:
    split_manifest = json.loads(Path(COMPANY_SPLIT_PATH).read_text())
    system_prompt = Path(FIN_QA_QUESTION_GENERATION_SYSTEM_PROMPT_PATH).read_text()
    verification_prompt = Path(FIN_QA_QUESTION_VERIFICATION_SYSTEM_PROMPT_PATH).read_text()
    client = OpenAI(base_url="http://localhost:30000/v1", api_key="dummy")

    splits = VALID_SPLITS if target_split is None else (target_split,)
    invalid = [split for split in splits if split not in VALID_SPLITS]
    if invalid:
        print(f"Unknown split(s) {', '.join(invalid)}. Choose from train, val, test.")
        return

    for split in splits:

        companies = split_manifest.get(split, [])
        if not companies:
            print(f"{split}: no companies in split file")
            continue

        rows = []
        total_companies = len(companies)
        with ThreadPoolExecutor(max_workers=COMPANY_WORKERS) as executor:
            futures = []
            for index, company in enumerate(companies, 1):
                print(f"{split}: [{index}/{total_companies}] {company}", flush=True)
                futures.append((company, executor.submit(process_company, client, system_prompt, verification_prompt, company, split)))
            for _, future in futures:
                rows.extend(future.result())

        finalize_split(split, rows, OUTPUT_PATHS[split])


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
