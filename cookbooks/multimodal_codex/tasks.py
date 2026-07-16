"""Procedural visual task generators for multimodal RL training.

Each generator returns a dict row suitable for DatasetRegistry. Images are NOT
stored — only the deterministic parameters needed to regenerate them. The flow
function recreates the image from these params at rollout time.

Task types:
- chart_reading: bar chart with labeled values, ask for a specific bar's value
- shape_counting: random geometric shapes, ask how many of a target shape
- table_reading: rendered table image, ask for a specific cell value
"""

from __future__ import annotations

import random


def generate_chart_reading_row(rng: random.Random, idx: int) -> dict:
    """Generate params for a bar chart reading task."""
    n_cats = rng.randint(3, 6)
    categories = [chr(ord("A") + i) for i in range(n_cats)]
    values = [rng.randint(10, 95) for _ in categories]
    target_idx = rng.randint(0, n_cats - 1)

    return {
        "uid": f"chart_{idx:06d}",
        "task_type": "chart_reading",
        "categories": categories,
        "values": values,
        "target_idx": target_idx,
        "ground_truth": str(values[target_idx]),
        "question": f"What is the exact value for category '{categories[target_idx]}' in the bar chart? Answer with only the number.",
    }


def generate_shape_count_row(rng: random.Random, idx: int) -> dict:
    """Generate params for a shape counting task."""
    n_shapes = rng.randint(4, 12)
    shape_types = ["circle", "square", "triangle"]
    placements = []
    counts = {"circle": 0, "square": 0, "triangle": 0}

    for _ in range(n_shapes):
        shape = rng.choice(shape_types)
        x = rng.randint(20, 180)
        y = rng.randint(20, 180)
        size = rng.randint(8, 22)
        color_idx = rng.randint(0, 4)
        placements.append({"shape": shape, "x": x, "y": y, "size": size, "color_idx": color_idx})
        counts[shape] += 1

    nonzero = [s for s, c in counts.items() if c > 0]
    target_shape = rng.choice(nonzero)

    return {
        "uid": f"shape_{idx:06d}",
        "task_type": "shape_counting",
        "placements": placements,
        "target_shape": target_shape,
        "ground_truth": str(counts[target_shape]),
        "question": f"Count the number of {target_shape}s in the image. Answer with only the number.",
    }


def generate_table_read_row(rng: random.Random, idx: int) -> dict:
    """Generate params for a table reading task."""
    rows = rng.randint(3, 6)
    cols = rng.randint(2, 4)
    headers = [f"Col{i + 1}" for i in range(cols)]
    data = [[rng.randint(1, 999) for _ in range(cols)] for _ in range(rows)]

    target_row = rng.randint(0, rows - 1)
    target_col = rng.randint(0, cols - 1)

    return {
        "uid": f"table_{idx:06d}",
        "task_type": "table_reading",
        "headers": headers,
        "table_data": data,
        "target_row": target_row,
        "target_col": target_col,
        "ground_truth": str(data[target_row][target_col]),
        "question": f"Look at the table. What is the value in Row {target_row + 1}, {headers[target_col]}? Answer with only the number.",
    }


GENERATORS = [generate_chart_reading_row, generate_shape_count_row, generate_table_read_row]
