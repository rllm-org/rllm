# Standard imports
import io
import json
import os
import re
import sqlite3
import threading
import warnings

import pandas as pd
from asteval import Interpreter
from pandas.api.types import is_numeric_dtype

from rllm.tools.tool_base import Tool

from .constants import TABLES_CLEANED_ALL_COMPANIES_FILE_NAME, TABLES_ROOT

# =============================================================================
# GLOBAL CACHES (populated once at module import)
# =============================================================================

_DB_CONN: sqlite3.Connection | None = None
_TABLE_INFO_CACHE: dict[tuple, str] = {}  # (company, table) -> JSON string
_COMPANY_TABLES: dict[str, list] = {}  # company -> [table1, table2, ...]
_SQL_NAMES: dict[tuple, str] = {}  # (company, table) -> sql_table_name
_TABLE_NAME_MAP: dict[tuple, str] = {}  # (company, table_lower) -> original_table_name
_PRELOADED: bool = False
_DB_LOCK = threading.Lock()  # Lock for thread-safe SQL queries


def _normalize_company(company_name: str) -> str:
    """Normalize company name to lowercase for case-insensitive lookup."""
    return company_name.lower().strip()


def _normalize_table(company: str, table_name: str) -> str | None:
    """Look up original table name from case-insensitive input. Returns None if not found."""
    table_name = table_name.strip()
    if table_name.endswith(".json"):
        table_name = table_name[:-5]
    return _TABLE_NAME_MAP.get((company, table_name.lower()))


def _get_sql_name(company: str, table: str) -> str:
    """Generate SQL table name. Same logic as before for compatibility."""
    cleaned = os.path.splitext(os.path.basename(table))[0]
    return f"{company}_{cleaned}".replace("-", "_").replace(" ", "_")


def _safe_json_value(val):
    """Convert value to JSON-serializable form."""
    if pd.isna(val):
        return None
    if hasattr(val, "isoformat"):  # datetime/Timestamp
        return str(val)
    return val


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns have valid non-empty names."""
    new_cols = []
    for i, col in enumerate(df.columns):
        if not col or str(col).strip() == "":
            new_cols.append(f"col_{i}")
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def _compute_table_info(table: str, metadata: dict, df: pd.DataFrame) -> str:
    """Pre-compute get_table_info response. Same logic as original."""
    table_payload = dict(metadata)
    table_payload.pop("table", None)

    if df.empty or len(df.columns) == 0:
        table_payload["column_names"] = []
        table_payload["column_dtypes"] = {}
        table_payload["unique_vals_per_col"] = {}
        return json.dumps(table_payload, indent=0).replace("\n", "")

    table_payload["column_names"] = df.columns.tolist()
    table_payload["column_dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    # Index info
    index_info = None
    if len(df.columns) > 0:
        index_column = df.columns[0]
        index_series = df[index_column]
        if not is_numeric_dtype(index_series):
            sample_values = index_series.dropna().astype(str).tolist()
            if sample_values:
                index_info = {"name": index_column, "values": sample_values}

    # Filter numeric columns for unique_vals_per_col
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


def _preload_company(company: str) -> tuple:
    """Load all tables for a company into caches. Returns (loaded, skipped) counts."""
    global _DB_CONN

    metadata_path = os.path.join(TABLES_ROOT, company, TABLES_CLEANED_ALL_COMPANIES_FILE_NAME)
    if not os.path.exists(metadata_path):
        _COMPANY_TABLES[company] = []
        return 0, 0

    try:
        with open(metadata_path) as f:
            all_tables = json.load(f)
    except json.JSONDecodeError as e:
        warnings.warn(
            f"Skipping company '{company}': malformed JSON in metadata file - {e}",
            stacklevel=2,
        )
        _COMPANY_TABLES[company] = []
        return 0, 0

    loaded, skipped = 0, 0
    valid_tables = []

    for table, info in all_tables.items():
        try:
            table_json = info.get("table", "{}")
            df = pd.read_json(io.StringIO(table_json), convert_dates=False)
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


def _preload_all():
    """Pre-load all data at module import. Called once."""
    global _DB_CONN, _PRELOADED

    if _PRELOADED:
        return

    _DB_CONN = sqlite3.connect(":memory:", check_same_thread=False)

    # Get all company directories
    if not os.path.exists(TABLES_ROOT):
        _PRELOADED = True
        return

    companies = [name for name in os.listdir(TABLES_ROOT) if os.path.isdir(os.path.join(TABLES_ROOT, name))]

    total_loaded, total_skipped = 0, 0
    for company in companies:
        loaded, skipped = _preload_company(company)
        total_loaded += loaded
        total_skipped += skipped

    _PRELOADED = True


# Pre-load everything at module import
_preload_all()


class GetTableNames(Tool):
    """A tool to get a list of possible tables for a given company."""

    NAME = "get_table_names"
    DESCRIPTION = """
A tool to get a list of possible tables for a given company.

Arguments:
    company_name: The name of the company to get the table names for.
Returns:
    A list of possible tables to look up described by their complete name.

Example:
    Input  -> {"company_name": "apple"}
    Output -> ["us_gaap_DisaggregationOfRevenueTableTextBlock", 'us_gaap_ScheduleOfComponentsOfIncomeTaxExpenseBenefitTableTextBlock', ...]
    """

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.get_table_names)
        # No filesystem operations - data already pre-loaded into _COMPANY_TABLES

    def get_table_names(self, company_name: str) -> list[str] | str:
        """Return table identifiers available for the provided company."""
        company = _normalize_company(company_name)
        if company not in _COMPANY_TABLES:
            return f"Error: Company name {company_name} not found, use a valid company name."
        # Only return tables that can actually be queried (loaded into SQLite)
        return [t for t in _COMPANY_TABLES[company] if (company, t) in _SQL_NAMES]


class GetTableInfo(Tool):
    """
    A tool to get information about a table.
    """

    NAME = "get_table_info"
    DESCRIPTION = """
Return metadata for a table belonging to a company.

Arguments:
    company_name: The name of the company to get the information for.
    table_name: Table identifier (filename without extension) returned by get_table_names.
Returns:
    A JSON string containing:
        - description: Description of the table
        - table_name: Name of the table
        - column_names: List of column names
        - column_dtypes: Mapping of column name to pandas dtype.
        - unique_vals_per_col: Sample VALUES from each column (use in WHERE filters, NOT as column names in SELECT).
        - index: When present, contains `name` (the first column) and `values` (row labels such as "Less imputed interest").
        Note: Many numeric cells are stored as strings (e.g., "1,234"), so remove punctuation before casting.

Example:
    Input  -> {"company_name": "apple", "table_name": "us_gaap_DisaggregationOfRevenueTableTextBlock"}
    Output -> "{\"description\": ..., \"column_names\": [...], ...}"
    """

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.get_table_info)
        # No filesystem operations - data already pre-loaded into _TABLE_INFO_CACHE

    def get_table_info(self, company_name: str, table_name: str) -> str:
        """Return metadata (as a JSON string) for the requested table."""
        company = _normalize_company(company_name)
        if company not in _COMPANY_TABLES:
            return f'Error: Company "{company_name}" not found.'
        table = _normalize_table(company, table_name)
        if table is None:
            return f'Error: Table "{table_name}" not found for company "{company_name}". Use get_table_names for full list.'
        return _TABLE_INFO_CACHE[(company, table)]


class SQLQuery(Tool):
    """
    A tool to query a table.

    Uses a global shared database to be efficient.
    """

    NAME = "sql_query"
    DESCRIPTION = """
Given a table name, and a SQL query, use SQLite to process the query over the table, and return the result. 
Provide your queries in SQLite compatible format. 
If the query is for the whole table/whole columns without filters, the query is too inefficient, and you will get an error.

Args:
    company_name: Name of the company provided by the user
    table_name: Name of the table to query
    query: SQL query to execute on the table 

Returns:
    str: Result of the query in JSON format

Example:
    Input  -> {"company_name": "apple", "table_name": "us_gaap_ScheduleOfMaturitiesOfLongTermDebtTableTextBlock", "query": "SELECT debt_category, amount FROM us_gaap_ScheduleOfMaturitiesOfLongTermDebtTableTextBlock WHERE amount != '' LIMIT 3"}
    Output -> "[{\"debt_category\": ...}, ...]"
    """

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.sql_query)

    def sql_query(self, company_name: str, table_name: str, query: str) -> str:
        if not query or not query.strip():
            return "Error : query must not be empty."

        # Normalize company and table names (case-insensitive)
        company = _normalize_company(company_name)
        table = _normalize_table(company, table_name)
        if table is None:
            return f"Error: table {table_name} for company {company_name} could not be found."

        query_upper = re.sub(r"(\\r|\\n|\\t|[\r\n\t])+", " ", query).upper()
        if "SELECT *" in query_upper:
            return f'Error : "SELECT *" is not allowed. Please list columns explicitly, e.g., SELECT column1, column2 FROM {table} LIMIT 5.'

        # Require at least one filter or limiting clause to keep queries efficient
        sql_filters = (
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
        if not any(clause in query_upper for clause in sql_filters):
            return "Error: Query needs a filter (WHERE, LIMIT, GROUP BY) or aggregate (COUNT, SUM, AVG, MIN, MAX)."

        # Check for missing FROM clause
        if "SELECT" in query_upper and "FROM" not in query_upper:
            return f"Error: Missing FROM clause. Use: SELECT ... FROM {table_name} ..."

        # Get pre-loaded SQL table name
        sql_name = _SQL_NAMES.get((company, table))
        if not sql_name:
            cleaned_table_name = os.path.splitext(os.path.basename(table_name))[0]
            return f"Error: table {cleaned_table_name} for company {company_name} could not be loaded."

        # QUOTED for SQLite compatibility - fixes table names starting with digits (e.g., "3m")
        quoted_sql_name = f'"{sql_name}"'

        # SINGLE REGEX handles: backticks, quotes, case-insensitivity,
        # column collision, and string literal issues (only replaces after FROM/JOIN)
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

        # Detect fake columns (SQLite returns column name as value for non-existent double-quoted columns)
        if not resp.empty:
            cursor = _DB_CONN.cursor()
            cursor.execute(f"PRAGMA table_info({quoted_sql_name})")
            valid_cols = {row[1].lower() for row in cursor.fetchall()}

            for col in resp.columns:
                col_clean = col.strip("\"'`")
                # If column value equals column name, it's a fake column echoed back
                if col_clean.lower() not in valid_cols:
                    if len(resp) > 0 and str(resp[col].iloc[0]) == col_clean:
                        return f'Error: Column "{col_clean}" not found. Available columns are: {", ".join(sorted(valid_cols))}'

        return resp.to_json(orient="records")


class Calculator(Tool):
    """
    A tool to safely evaluate a mathematical expression.
    """

    NAME = "calculator"
    DESCRIPTION = """
Evaluate a mathematical expression and return the numeric result.

Arguments:
    expression: Expression string to evaluate.

Returns:
    float: Numeric result.

Example:
    Input -> {"expression": "(12 + 8) / 5"}
    Output -> 4.0
"""

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description, function=self.calculator)

    def calculator(self, expression: str) -> float | str:
        if not isinstance(expression, str):
            return "Error: Input expression must be a string."

        expr = expression

        # 1. Strip whitespace (fixes "unexpected indent" - 107 errors in training)
        expr = expr.strip()

        # 2. Normalize newlines (fixes silent wrong results: "100\n+50" returned 50!)
        expr = expr.replace("\n", " ").replace("\r", " ")

        # 3. CRITICAL: Convert ^ to ** (fixes XOR vs power: "2^3" returned 1 instead of 8!)
        expr = expr.replace("^", "**")

        # 4. Remove currency symbols
        expr = expr.replace("$", "").replace("€", "").replace("£", "")

        # 5. Normalize whitespace (non-breaking space U+00A0)
        expr = expr.replace("\u00a0", " ")

        # 6. Normalize dashes to minus (en-dash, em-dash, unicode minus sign)
        expr = expr.replace("–", "-").replace("—", "-").replace("−", "-")

        # 7. Unicode math symbols (fullwidth parens, multiply, divide)
        for old, new in {"（": "(", "）": ")", "×": "*", "÷": "/"}.items():
            expr = expr.replace(old, new)

        # 8. Fullwidth digits (Japanese/Chinese input methods)
        for i, fw in enumerate("０１２３４５６７８９"):
            expr = expr.replace(fw, str(i))

        # 9. Percentage to decimal (no \s* before % - would break modulo operator!)
        expr = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", expr)

        # 10. Thousand separators (safe pattern - preserves function calls like max(1, 2))
        expr = re.sub(r"\d{1,3}(?:,\d{3})+", lambda m: m.group(0).replace(",", ""), expr)

        try:
            aeval = Interpreter()
            result = aeval.eval(expr)
            return float(result)
        except Exception as e:
            return f"Error evaluating expression: '{expression}'. Details: {e}"
