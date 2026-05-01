"""FinQA tool implementations.

Four tools the agent calls to answer SEC-10K questions:

* ``get_table_names``  — list tables available for a company
* ``get_table_info``   — column/dtype/sample-value metadata for a table
* ``sql_query``        — run filtered SQL on the in-memory SQLite store
* ``calculator``       — evaluate a numeric expression safely (asteval)

All company tables are pre-loaded into a process-wide SQLite ``:memory:``
DB at module import; the tools just look up cached metadata or run a
read-only query under a thread lock. The cookbook ships these as plain
Python callables (no ``Tool`` base class) plus the OpenAI function-call
specs ``TOOL_SPECS`` so the flow can pass them straight to
``client.chat.completions.create(tools=TOOL_SPECS)``.
"""

from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import threading
import warnings

import pandas as pd
from asteval import Interpreter
from finqa_constants import TABLES_CLEANED_ALL_COMPANIES_FILE_NAME, TABLES_ROOT
from pandas.api.types import is_numeric_dtype

# ---------------------------------------------------------------------------
# Process-wide caches populated once at module import
# ---------------------------------------------------------------------------

_DB_CONN: sqlite3.Connection | None = None
_TABLE_INFO_CACHE: dict[tuple, str] = {}  # (company, table)        -> JSON string
_COMPANY_TABLES: dict[str, list] = {}  # company                 -> [table1, table2, ...]
_SQL_NAMES: dict[tuple, str] = {}  # (company, table)        -> sql_table_name
_TABLE_NAME_MAP: dict[tuple, str] = {}  # (company, table_lower)  -> original_table_name
_PRELOADED: bool = False
_DB_LOCK = threading.Lock()


def _normalize_company(company_name: str) -> str:
    return company_name.lower().strip()


def _normalize_table(company: str, table_name: str) -> str | None:
    table_name = table_name.strip()
    if table_name.endswith(".json"):
        table_name = table_name[:-5]
    return _TABLE_NAME_MAP.get((company, table_name.lower()))


def _get_sql_name(company: str, table: str) -> str:
    cleaned = os.path.splitext(os.path.basename(table))[0]
    return f"{company}_{cleaned}".replace("-", "_").replace(" ", "_")


def _safe_json_value(val):
    if pd.isna(val):
        return None
    if hasattr(val, "isoformat"):
        return str(val)
    return val


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for i, col in enumerate(df.columns):
        if not col or str(col).strip() == "":
            new_cols.append(f"col_{i}")
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def _compute_table_info(table: str, metadata: dict, df: pd.DataFrame) -> str:
    table_payload = dict(metadata)
    table_payload.pop("table", None)

    if df.empty or len(df.columns) == 0:
        table_payload["column_names"] = []
        table_payload["column_dtypes"] = {}
        table_payload["unique_vals_per_col"] = {}
        return json.dumps(table_payload, indent=0).replace("\n", "")

    table_payload["column_names"] = df.columns.tolist()
    table_payload["column_dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    index_info = None
    if len(df.columns) > 0:
        index_column = df.columns[0]
        index_series = df[index_column]
        if not is_numeric_dtype(index_series):
            sample_values = index_series.dropna().astype(str).tolist()
            if sample_values:
                index_info = {"name": index_column, "values": sample_values}

    cols_to_drop = []
    for col in df.columns.tolist()[1:]:
        values = df[col].tolist()[1:]
        cleaned = ["".join(ch for ch in str(v) if ch.isalnum()).strip() for v in values]
        if all(val.isnumeric() or len(val) == 0 for val in cleaned):
            cols_to_drop.append(col)

    df_filtered = df.drop(columns=cols_to_drop)
    table_payload["unique_vals_per_col"] = {col: [_safe_json_value(v) for v in df_filtered[col].dropna().unique().tolist()] for col in df_filtered.columns}

    if index_info:
        name = index_info["name"]
        table_payload["index"] = {
            "name": name,
            "values": table_payload["unique_vals_per_col"].get(name, index_info["values"]),
        }

    return json.dumps(table_payload, indent=0).replace("\n", "")


def _preload_company(company: str) -> tuple[int, int]:
    global _DB_CONN

    metadata_path = os.path.join(TABLES_ROOT, company, TABLES_CLEANED_ALL_COMPANIES_FILE_NAME)
    if not os.path.exists(metadata_path):
        _COMPANY_TABLES[company] = []
        return 0, 0

    try:
        with open(metadata_path) as f:
            all_tables = json.load(f)
    except json.JSONDecodeError as e:
        warnings.warn(f"Skipping company '{company}': malformed JSON in metadata file - {e}", stacklevel=2)
        _COMPANY_TABLES[company] = []
        return 0, 0

    loaded, skipped = 0, 0
    valid_tables = []
    for table, info in all_tables.items():
        try:
            df = pd.read_json(io.StringIO(info.get("table", "{}")), convert_dates=False)
            df = _sanitize_columns(df)
            _TABLE_INFO_CACHE[(company, table)] = _compute_table_info(table, info, df)
            _TABLE_NAME_MAP[(company, table.lower())] = table
            valid_tables.append(table)

            if df.empty or len(df.columns) == 0:
                skipped += 1
                continue

            sql_name = _get_sql_name(company, table)
            _SQL_NAMES[(company, table)] = sql_name
            df.to_sql(sql_name, _DB_CONN, index=False, if_exists="replace")
            loaded += 1
        except ValueError as e:
            warnings.warn(f"Skipping table '{table}' for company '{company}': {e}", stacklevel=2)

    _COMPANY_TABLES[company] = valid_tables
    return loaded, skipped


def _preload_all() -> None:
    """Load every company's tables into the in-memory SQLite store. Idempotent."""
    global _DB_CONN, _PRELOADED

    if _PRELOADED:
        return

    _DB_CONN = sqlite3.connect(":memory:", check_same_thread=False)

    if not os.path.exists(TABLES_ROOT):
        # Tools still load and tests still run; agents will get clean
        # "company not found" errors when called.
        _PRELOADED = True
        return

    companies = [n for n in os.listdir(TABLES_ROOT) if os.path.isdir(os.path.join(TABLES_ROOT, n))]
    for company in companies:
        _preload_company(company)
    _PRELOADED = True


# Pre-load at import. Set FINQA_SKIP_PRELOAD=1 to skip (used by unit tests).
if not os.getenv("FINQA_SKIP_PRELOAD"):
    _preload_all()


# ---------------------------------------------------------------------------
# Tool implementations (plain functions)
# ---------------------------------------------------------------------------


def get_table_names(company_name: str) -> list[str] | str:
    """Return the queryable table identifiers for a company."""
    company = _normalize_company(company_name)
    if company not in _COMPANY_TABLES:
        return f"Error: Company name {company_name} not found, use a valid company name."
    return [t for t in _COMPANY_TABLES[company] if (company, t) in _SQL_NAMES]


def get_table_info(company_name: str, table_name: str) -> str:
    """Return JSON metadata (columns, dtypes, sample values) for a table."""
    company = _normalize_company(company_name)
    if company not in _COMPANY_TABLES:
        return f'Error: Company "{company_name}" not found.'
    table = _normalize_table(company, table_name)
    if table is None:
        return f'Error: Table "{table_name}" not found for company "{company_name}". Use get_table_names for full list.'
    return _TABLE_INFO_CACHE[(company, table)]


_SQL_FILTER_TOKENS = (
    " WHERE ",
    " HAVING ",
    " GROUP BY ",
    " ORDER BY ",
    " LIMIT ",
    " OFFSET ",
    " IN ",
    " NOT IN ",
    " EXISTS ",
    " NOT EXISTS ",
    " LIKE ",
    " BETWEEN ",
    " FILTER ",
    " COUNT(",
    " MAX(",
    " MIN(",
    " SUM(",
    " AVG(",
)


def sql_query(company_name: str, table_name: str, query: str) -> str:
    """Run a filtered SQLite query against a pre-loaded company table."""
    if not query or not query.strip():
        return "Error : query must not be empty."

    company = _normalize_company(company_name)
    table = _normalize_table(company, table_name)
    if table is None:
        return f"Error: table {table_name} for company {company_name} could not be found."

    query_upper = re.sub(r"(\\r|\\n|\\t|[\r\n\t])+", " ", query).upper()
    if "SELECT *" in query_upper:
        return f'Error : "SELECT *" is not allowed. Please list columns explicitly, e.g., SELECT column1, column2 FROM {table} LIMIT 5.'
    if not any(clause in query_upper for clause in _SQL_FILTER_TOKENS):
        return "Error: Query needs a filter (WHERE, LIMIT, GROUP BY) or aggregate (COUNT, SUM, AVG, MIN, MAX)."
    if "SELECT" in query_upper and "FROM" not in query_upper:
        return f"Error: Missing FROM clause. Use: SELECT ... FROM {table_name} ..."

    sql_name = _SQL_NAMES.get((company, table))
    if not sql_name:
        cleaned_table_name = os.path.splitext(os.path.basename(table_name))[0]
        return f"Error: table {cleaned_table_name} for company {company_name} could not be loaded."

    quoted_sql_name = f'"{sql_name}"'
    pattern = rf'(FROM|JOIN)\s+[`"\']?{re.escape(table_name)}[`"\']?(?=\s|$|;|,|\))'
    query_for_execution = re.sub(pattern, rf"\1 {quoted_sql_name}", query, flags=re.IGNORECASE)

    try:
        with _DB_LOCK:
            resp = pd.read_sql_query(query_for_execution, _DB_CONN)
    except Exception as e:
        err_msg = str(e)
        if "no such column" in err_msg.lower():
            try:
                cursor = _DB_CONN.cursor()
                cursor.execute(f"PRAGMA table_info({quoted_sql_name})")
                columns = [row[1] for row in cursor.fetchall()]
                return "Error : Column not found in table. Available columns are: " + ", ".join(columns)
            except Exception:
                pass
        return f"Error: {err_msg}"

    if not resp.empty:
        cursor = _DB_CONN.cursor()
        cursor.execute(f"PRAGMA table_info({quoted_sql_name})")
        valid_cols = {row[1].lower() for row in cursor.fetchall()}
        for col in resp.columns:
            col_clean = col.strip("\"'`")
            if col_clean.lower() not in valid_cols and len(resp) > 0 and str(resp[col].iloc[0]) == col_clean:
                return f'Error: Column "{col_clean}" not found. Available columns are: {", ".join(sorted(valid_cols))}'

    return resp.to_json(orient="records")


def calculator(expression: str) -> float | str:
    """Evaluate a numeric expression safely (asteval). Handles common LLM quirks
    like fullwidth digits, $/€/£ currency symbols, %, en/em-dashes, and ^ for power."""
    if not isinstance(expression, str):
        return "Error: Input expression must be a string."

    expr = expression.strip()
    expr = expr.replace("\n", " ").replace("\r", " ")
    expr = expr.replace("^", "**")  # XOR -> power
    expr = expr.replace("$", "").replace("€", "").replace("£", "")
    expr = expr.replace("\u00a0", " ")
    expr = expr.replace("–", "-").replace("—", "-").replace("−", "-")
    for old, new in {"(": "(", ")": ")", "×": "*", "÷": "/"}.items():
        expr = expr.replace(old, new)
    for i, fw in enumerate("０１２３４５６７８９"):
        expr = expr.replace(fw, str(i))
    expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr)
    expr = re.sub(r"\d{1,3}(?:,\d{3})+", lambda m: m.group(0).replace(",", ""), expr)

    try:
        result = Interpreter().eval(expr)
        return float(result)
    except Exception as e:
        return f"Error evaluating expression: '{expression}'. Details: {e}"


# ---------------------------------------------------------------------------
# OpenAI function-call specs
# ---------------------------------------------------------------------------

TOOL_FNS = {
    "get_table_names": get_table_names,
    "get_table_info": get_table_info,
    "sql_query": sql_query,
    "calculator": calculator,
}


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_table_names",
            "description": "List the tables available for a given company. Returns table identifiers (no extension) suitable for use with get_table_info / sql_query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "Company name (case-insensitive)."},
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_info",
            "description": "Return metadata (description, column names, dtypes, sample values) for a specific table. Use the identifier returned by get_table_names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                    "table_name": {"type": "string"},
                },
                "required": ["company_name", "table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": (
                "Execute a filtered SQLite query against a pre-loaded company table. "
                "Must include a WHERE / LIMIT / GROUP BY / ORDER BY clause or an aggregate "
                "(COUNT/MAX/MIN/SUM/AVG). SELECT * is rejected — list columns explicitly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                    "table_name": {"type": "string"},
                    "query": {"type": "string", "description": "SQLite query string."},
                },
                "required": ["company_name", "table_name", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate a numeric expression and return the float result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression."},
                },
                "required": ["expression"],
            },
        },
    },
]
