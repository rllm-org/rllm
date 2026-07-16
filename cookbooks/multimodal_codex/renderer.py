"""PIL-only image rendering for the multimodal_codex cookbook.

Each renderer takes a row dict produced by ``tasks.py`` generators and returns
PNG bytes. Kept dependency-light (no matplotlib) so the pod venv stays lean.

Dispatch entry: ``_render_image_from_params(row) -> bytes``.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

# 5-color palette (matches tasks.py color_idx range 0..4).
_PALETTE = [
    (66, 133, 244),   # blue
    (234, 67, 53),    # red
    (52, 168, 83),    # green
    (251, 188, 5),    # yellow
    (156, 39, 176),   # purple
]

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREY = (180, 180, 180)


def _font(size: int) -> ImageFont.ImageFont:
    """Prefer default bitmap font — always available, no filesystem lookup."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older Pillow: load_default() takes no size arg
        return ImageFont.load_default()


def _render_chart(row: dict) -> bytes:
    """Bar chart: N labeled bars with values on top."""
    categories = row["categories"]
    values = row["values"]
    n = len(categories)

    img_w, img_h = 480, 320
    margin_l, margin_r, margin_t, margin_b = 50, 20, 30, 50
    plot_w = img_w - margin_l - margin_r
    plot_h = img_h - margin_t - margin_b

    img = Image.new("RGB", (img_w, img_h), _WHITE)
    draw = ImageDraw.Draw(img)

    # Axes
    draw.line([(margin_l, margin_t), (margin_l, img_h - margin_b)], fill=_BLACK, width=2)
    draw.line([(margin_l, img_h - margin_b), (img_w - margin_r, img_h - margin_b)], fill=_BLACK, width=2)

    v_max = max(values) if values else 1
    v_max = max(v_max, 1)  # avoid /0
    bar_slot = plot_w / n
    bar_w = int(bar_slot * 0.6)
    baseline_y = img_h - margin_b

    label_font = _font(14)
    value_font = _font(14)

    for i, (cat, val) in enumerate(zip(categories, values)):
        cx = margin_l + int((i + 0.5) * bar_slot)
        bar_h = int((val / v_max) * (plot_h - 10))
        x0 = cx - bar_w // 2
        y0 = baseline_y - bar_h
        x1 = cx + bar_w // 2
        y1 = baseline_y
        color = _PALETTE[i % len(_PALETTE)]
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=_BLACK)
        draw.text((cx - 8, baseline_y + 6), str(cat), fill=_BLACK, font=label_font)
        draw.text((cx - 10, y0 - 18), str(val), fill=_BLACK, font=value_font)

    return _to_png_bytes(img)


def _render_shapes(row: dict) -> bytes:
    """Random geometric shapes on a canvas."""
    img_w, img_h = 240, 240
    img = Image.new("RGB", (img_w, img_h), _WHITE)
    draw = ImageDraw.Draw(img)

    for p in row["placements"]:
        cx, cy, size = p["x"], p["y"], p["size"]
        color = _PALETTE[p["color_idx"] % len(_PALETTE)]
        shape = p["shape"]
        if shape == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color, outline=_BLACK)
        elif shape == "square":
            draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color, outline=_BLACK)
        elif shape == "triangle":
            draw.polygon(
                [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)],
                fill=color,
                outline=_BLACK,
            )

    return _to_png_bytes(img)


def _render_table(row: dict) -> bytes:
    """Simple grid: header row + data rows, cells sized to fit content."""
    headers = row["headers"]
    data = row["table_data"]
    n_rows = len(data) + 1  # + header
    n_cols = len(headers)

    cell_w, cell_h = 80, 34
    img_w = cell_w * n_cols + 2
    img_h = cell_h * n_rows + 2
    img = Image.new("RGB", (img_w, img_h), _WHITE)
    draw = ImageDraw.Draw(img)

    header_font = _font(14)
    body_font = _font(14)

    # Header row (grey background)
    draw.rectangle([1, 1, img_w - 1, cell_h], fill=(230, 230, 230), outline=_BLACK)
    for c, h in enumerate(headers):
        x = c * cell_w + cell_w // 2 - 15
        y = cell_h // 2 - 8
        draw.text((x, y), str(h), fill=_BLACK, font=header_font)

    # Body rows
    for r, row_vals in enumerate(data):
        for c, val in enumerate(row_vals):
            x = c * cell_w + cell_w // 2 - 10
            y = (r + 1) * cell_h + cell_h // 2 - 8
            draw.text((x, y), str(val), fill=_BLACK, font=body_font)

    # Grid lines
    for c in range(n_cols + 1):
        x = c * cell_w + 1
        draw.line([(x, 0), (x, img_h)], fill=_GREY, width=1)
    for r in range(n_rows + 1):
        y = r * cell_h + 1
        draw.line([(0, y), (img_w, y)], fill=_GREY, width=1)

    return _to_png_bytes(img)


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_DISPATCH = {
    "chart_reading": _render_chart,
    "shape_counting": _render_shapes,
    "table_reading": _render_table,
}


def _render_image_from_params(row: dict) -> bytes:
    """Dispatch to the correct renderer based on ``row['task_type']``."""
    task_type = row.get("task_type")
    fn = _DISPATCH.get(task_type)
    if fn is None:
        raise ValueError(f"Unknown task_type: {task_type!r} (expected one of {list(_DISPATCH)})")
    return fn(row)
