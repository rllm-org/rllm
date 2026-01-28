import json
import random
from pathlib import Path

from projects.finqa.constants import (
    COMPANY_SPLIT_PATH,
    TABLES_CLEANED_ALL_COMPANIES_FILE_NAME,
    TABLES_ROOT,
)


def load_companies(tables_root: Path) -> list[str]:
    if not tables_root.is_dir():
        raise FileNotFoundError(f"Tables directory not found: {tables_root}")
    companies = sorted(entry.name for entry in tables_root.iterdir() if entry.is_dir())
    if not companies:
        raise ValueError(f"No company folders found in {tables_root}")
    return companies


def split_companies(companies: list[str]) -> dict[str, list[str]]:
    shuffled = companies.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * 0.8)
    val_count = int(total * 0.1)

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count :]

    return {"train": train, "val": val, "test": test}


def count_tables(company_dir: Path) -> int:
    tables_file = company_dir / TABLES_CLEANED_ALL_COMPANIES_FILE_NAME
    if not tables_file.is_file():
        raise FileNotFoundError(
            f"Missing tables file for company {company_dir.name}: {tables_file}"
        )

    with tables_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    raise ValueError(f"Unsupported tables format in {tables_file}")


def write_metadata(
    splits: dict[str, list[str]],
    table_counts: dict[str, int],
    metadata_path: Path,
) -> None:
    lines = [f"Total companies: {len(table_counts)}"]
    for split_name in ("train", "val", "test"):
        names = splits.get(split_name, [])
        counts = [table_counts[name] for name in names]
        total_tables = sum(counts)
        avg_tables = (total_tables / len(counts)) if counts else 0.0
        lines.append(
            f"{split_name.title()}: {len(names)} companies | "
            f"{total_tables} tables | avg {avg_tables:.2f} tables/company"
        )

    metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tables_root = Path(TABLES_ROOT)
    companies = load_companies(tables_root)
    table_counts = {
        company: count_tables(tables_root / company) for company in companies
    }
    splits = split_companies(companies)

    split_path = Path(COMPANY_SPLIT_PATH)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metadata_path = split_path.with_name("metadata.txt")
    write_metadata(splits, table_counts, metadata_path)

    print(f"Wrote split file with {len(companies)} companies to {split_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
