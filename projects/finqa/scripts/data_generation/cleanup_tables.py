import argparse
import asyncio
import functools
import glob
import json
import os
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI
from tqdm.asyncio import tqdm

from projects.finqa.constants import TABLES_CLEANED_ALL_COMPANIES_FILE_NAME

COMPANY_CONCURRENCY = 2
FILE_CONCURRENCY = 5


def reformat_and_cleanup_table(table: str) -> tuple[str, str, str]:
    """
    Given html version of table, convert it into pandas dataframe.

    Args:
        table: HTML version of table

    Returns:
        tuple[str, str, str]: Pandas DataFrame in JSON format, title of the table, and description of the table
    """
    client = OpenAI(base_url="http://localhost:32000/v1", api_key="dummy")

    soup = BeautifulSoup(table, "html.parser")

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "headers": {"type": "array", "items": {"type": "string"}},
            "rows": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": {"type": "string"}},
            },
        },
        "required": ["title", "description", "headers", "rows"],
        "additionalProperties": False,
    }

    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        max_tokens=16384,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that parses HTML tables into structured data. Extract all rows and columns properly maintaining their relationships. Do not make up any new columns or rows, only use the ones present in the table. Do not add any new information. Make sure that all the column names are lowercase, and if there are spaces in the column names, replace them with underscores. The same goes for any other text within the table. Provide the title and descriptions in the same manner, with spaces replaced with underscores and in lowercase. If there are any dollar signs/millions/billions/trillions in the table, keep them, just reformat them in the manner I described.",
            },
            {
                "role": "user",
                "content": f"""
                Parse this HTML table into a structured format with headers and rows.
                Extract all rows and columns properly maintaining their relationships.
                 
                Please do not make up any new columns or rows, only use the ones that are present in the table.
                Please do not add in any new information, check your work before submitting.
                 
                If you cannot find information in the table, just return an empty dataframe.
                
                Return your response as a JSON object with this structure:
                {{
                    "title": "Title of the table",
                    "description": "Description of the table",
                    "headers": ["header1", "header2", ...],
                    "rows": [
                        {{"header1": "value1", "header2": "value2", ...}},
                        {{"header1": "value3", "header2": "value4", ...}},
                        ...
                    ]
                }}
                 
                Do not add any information in the Text block, only the JSON object.
                
                HTML Table:
                {soup.prettify()}
                """,
            },
        ],
        extra_body={"guided_json": schema},
    )

    text_block = response.choices[0].message.content

    json_obj = json.loads(text_block)

    df = pd.DataFrame(json_obj["rows"], columns=json_obj["headers"])
    # Retain duplicate-named columns by suffixing subsequent occurrences (col, col_2, col_3, ...)
    if df.columns.duplicated().any():
        seen_counts: dict[str, int] = {}
        renamed: list[str] = []
        for name in df.columns.tolist():
            count = seen_counts.get(name, 0) + 1
            seen_counts[name] = count
            renamed.append(name if count == 1 else f"{name}_{count}")
        df.columns = renamed
    title, description = json_obj["title"], json_obj["description"]
    return df.to_json(), title, description


async def process_file_async(file_path):
    # converts the blocking call to an async call
    loop = asyncio.get_event_loop()
    func = functools.partial(reformat_and_cleanup_table, Path(file_path).read_text())
    return await loop.run_in_executor(None, func)


async def context_handler(file_path, sem):
    try:
        async with sem:
            async with asyncio.timeout(100):
                df_json, title, description = await process_file_async(file_path)
                return {
                    "file": file_path,
                    "data": df_json,
                    "title": title,
                    "description": description,
                }
    except asyncio.TimeoutError:
        print(f"Request timed out after 100 seconds for {file_path}")
        return {"file": file_path, "error": "Request timed out after 100 seconds"}
    except Exception as e:
        print(f"Error for {file_path}: {e}")
        return {"file": file_path, "error": str(e)}


async def process_company(company_dir: str, file_sem: asyncio.Semaphore):
    company_name = os.path.basename(company_dir)
    file_paths = glob.glob(f"{company_dir}/*.txt")
    tasks = [context_handler(fp, file_sem) for fp in file_paths]
    if not tasks:
        return {"company": company_name, "processed": 0}
    results = await tqdm.gather(*tasks, desc=f"Processing {company_name}", total=len(tasks))

    company_summary = {}
    for item in results:
        file = item["file"]
        file_basename = os.path.basename(file)
        file_basename_cleaned = file_basename.replace("-", "_").removesuffix(".txt")
        file_basename_json = f"{file_basename_cleaned}.json"
        dirname = os.path.dirname(file)
        if "data" in item:
            df = pd.read_json(StringIO(item["data"]))
            cols = df.columns.tolist()
            unique_vals_per_col = {col: df[col].astype(str).unique().tolist() for col in cols}
            res = {
                "table": item["data"],
                "description": item["description"],
                "column_names": cols,
                "unique_vals_per_col": unique_vals_per_col,
                "company": company_name,
            }
            company_summary[file_basename_cleaned] = res
            Path(f"{dirname}/{file_basename_json}").write_text(item["data"])
        else:
            print(f"Skipping {file} because it does not have a table")

    # Write per-company summary for SQL tools consumption
    out_path = os.path.join(company_dir, TABLES_CLEANED_ALL_COMPANIES_FILE_NAME)
    with open(out_path, "w") as f:
        json.dump(company_summary, f)

    return {"company": company_name, "processed": len(company_summary)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up scraped 10-K HTML tables into structured JSON")
    parser.add_argument("--input_base_dir", type=str, required=True, help="Directory containing company subdirectories with .txt table files")
    args = parser.parse_args()

    company_dirs = [d for d in glob.glob(f"{args.input_base_dir}/*") if os.path.isdir(d)]
    print(f"Total companies : {len(company_dirs)}")

    async def _process_all_companies():
        company_sem = asyncio.Semaphore(COMPANY_CONCURRENCY)

        async def _run_one_company(cd: str):
            async with company_sem:
                file_sem = asyncio.Semaphore(FILE_CONCURRENCY)
                return await process_company(cd, file_sem)

        company_tasks = [_run_one_company(cd) for cd in company_dirs]
        return await asyncio.gather(*company_tasks)

    company_results = asyncio.run(_process_all_companies())
    with open("results-table-cleanup.json", "w") as f:
        json.dump(company_results, f)
