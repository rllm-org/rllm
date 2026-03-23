"""Table formatting utilities for CLI output."""

from __future__ import annotations


def format_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> str:
    """Format a simple text table.

    Args:
        headers: Column header strings.
        rows: List of row data (each row is a list of strings).
        col_widths: Optional explicit column widths. Auto-calculated if None.

    Returns:
        Formatted table string.
    """
    if col_widths is None:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else len(cell)
            parts.append(cell.ljust(w))
        return "  ".join(parts)

    lines = [fmt_row(headers), "  ".join("-" * w for w in col_widths)]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)
