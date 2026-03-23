"""HuggingFace dataset download and catalog loading utilities."""

from __future__ import annotations

import importlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# Default root for rllm datasets (overridden by RLLM_HOME env var)
_DATASETS_ROOT = os.path.join(os.environ.get("RLLM_HOME", os.path.expanduser("~/.rllm")), "datasets")


def load_dataset_catalog() -> dict:
    """Load the datasets.json catalog from the registry directory.

    Returns:
        dict: The full catalog with 'version' and 'datasets' keys.
    """
    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "registry", "datasets.json")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def load_agent_catalog() -> dict:
    """Load the agents.json catalog from the registry directory.

    Returns:
        dict: The full catalog with 'version' and 'agents' keys.
    """
    catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "registry", "agents.json")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def _load_transform(transform_path: str):
    """Load a transform function from a 'module:function' import path.

    Args:
        transform_path: Colon-separated import path (e.g., 'rllm.data.transforms:gpqa_diamond_transform').

    Returns:
        The transform function.
    """
    module_path, fn_name = transform_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def _remap_fields(row: dict, field_map: dict) -> dict:
    """Rename fields according to field_map. Preserves unmapped fields.

    Args:
        row: Original row dict.
        field_map: Mapping of source field name -> target field name.

    Returns:
        New dict with renamed fields.
    """
    new_row = {}
    for src_key, dst_key in field_map.items():
        if src_key in row:
            new_row[dst_key] = row[src_key]
    # Keep unmapped fields
    for key, val in row.items():
        if key not in field_map:
            new_row[key] = val
    return new_row


def _is_pil_image(obj: object) -> bool:
    """Check if an object is a PIL Image without importing PIL at module level."""
    try:
        from PIL import Image

        return isinstance(obj, Image.Image)
    except ImportError:
        return False


def _disable_image_decoding(hf_dataset):
    """Cast Image columns to decode=False so raw bytes are preserved.

    After this, Image columns return ``{"bytes": b"...", "path": ...}`` dicts
    instead of PIL objects, avoiding PIL decoding entirely.

    Handles both top-level ``Image`` features and ``Sequence(Image())``
    features (lists of images, e.g. zerobench's ``question_images_decoded``).

    Args:
        hf_dataset: A HuggingFace ``datasets.Dataset`` instance.

    Returns:
        The same dataset with Image columns cast to ``decode=False``.
    """
    from datasets import Image as HFImage
    from datasets import Sequence

    for col_name, feature in hf_dataset.features.items():
        if isinstance(feature, HFImage):
            hf_dataset = hf_dataset.cast_column(col_name, HFImage(decode=False))
        elif isinstance(feature, Sequence) and isinstance(feature.feature, HFImage):
            hf_dataset = hf_dataset.cast_column(col_name, Sequence(HFImage(decode=False)))
    return hf_dataset


def _flatten_image_dicts(data_list: list[dict]) -> list[dict]:
    """Extract raw bytes from HuggingFace image dicts.

    After ``decode=False``, image values are ``{"bytes": b"...", "path": ...}``
    dicts (or lists of them). This extracts just the ``bytes`` value:
    ``{"bytes": b"..."}`` → ``b"..."`` and
    ``[{"bytes": b"..."}, ...]`` → ``[b"...", ...]``.

    Non-image columns are left untouched.

    Args:
        data_list: List of row dicts.

    Returns:
        The same list with image dicts replaced by raw bytes.
    """
    if not data_list:
        return data_list

    def _is_image_dict(val):
        return isinstance(val, dict) and "bytes" in val

    # Detect columns with image dicts from the first row
    first_row = data_list[0]
    single_cols: list[str] = []
    list_cols: list[str] = []

    for col, val in first_row.items():
        if _is_image_dict(val):
            single_cols.append(col)
        elif isinstance(val, list) and val and _is_image_dict(val[0]):
            list_cols.append(col)

    if not single_cols and not list_cols:
        return data_list

    for row in data_list:
        for col in single_cols:
            val = row.get(col)
            if _is_image_dict(val):
                row[col] = val["bytes"]
            else:
                row[col] = None

        for col in list_cols:
            vals = row.get(col, [])
            if isinstance(vals, list):
                row[col] = [v["bytes"] if _is_image_dict(v) else None for v in vals]

    return data_list


def _extract_and_save_images(data_list: list[dict], name: str, split: str) -> list[dict]:
    """Scan rows for PIL Image columns, save to disk, and replace with relative paths.

    Images are saved to ``{DATASETS_ROOT}/{name}/images/{split}_{idx}_{col}.png``.
    The PIL objects are replaced with paths relative to ``{DATASETS_ROOT}/`` so
    that agent flows can resolve them at runtime.

    This is a no-op for datasets without PIL Image columns.

    .. note:: This is the legacy path kept for backward compatibility. New pulls
       use ``_disable_image_decoding`` + ``_flatten_image_dicts`` instead.

    Args:
        data_list: List of row dicts (potentially containing PIL Images).
        name: Dataset name (used in the image directory).
        split: Split name.

    Returns:
        The same list with PIL Image objects replaced by relative path strings.
    """
    if not data_list:
        return data_list

    # Detect image columns from the first row
    first_row = data_list[0]
    image_cols: list[str] = []
    list_image_cols: list[str] = []

    for col, val in first_row.items():
        if _is_pil_image(val):
            image_cols.append(col)
        elif isinstance(val, list) and val and _is_pil_image(val[0]):
            list_image_cols.append(col)

    if not image_cols and not list_image_cols:
        return data_list

    # Create image directory
    images_dir = os.path.join(_DATASETS_ROOT, name, "images")
    os.makedirs(images_dir, exist_ok=True)

    logger.info(f"  Extracting images for {name}/{split} (cols: {image_cols + list_image_cols})...")

    for idx, row in enumerate(data_list):
        for col in image_cols:
            img = row.get(col)
            if _is_pil_image(img):
                fname = f"{split}_{idx}_{col}.png"
                fpath = os.path.join(images_dir, fname)
                img.save(fpath)
                row[col] = os.path.join(name, "images", fname)
            else:
                row[col] = None

        for col in list_image_cols:
            imgs = row.get(col, [])
            paths = []
            if isinstance(imgs, list):
                for j, img in enumerate(imgs):
                    if _is_pil_image(img):
                        fname = f"{split}_{idx}_{col}_{j}.png"
                        fpath = os.path.join(images_dir, fname)
                        img.save(fpath)
                        paths.append(os.path.join(name, "images", fname))
            row[col] = paths

    logger.info(f"  Saved images to {images_dir}")
    return data_list


def _load_all_configs(source: str, split: str, hf_split: str | None = None) -> list[dict]:
    """Load all HF configs for a dataset and merge into a single list.

    Each row gets a 'language' column set to the config name (unless it
    already has one).

    Args:
        source: HuggingFace dataset path.
        split: The split to register under.
        hf_split: Optional override for the actual HF split to load.

    Returns:
        Merged list of row dicts with 'language' column.
    """
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(source)
    logger.info(f"  Aggregating {len(configs)} language configs...")
    all_rows: list[dict] = []

    for config_name in configs:
        try:
            ds = load_dataset(source, name=config_name, split=hf_split or split)
            ds = _disable_image_decoding(ds)
            for row in ds:
                row_dict = dict(row)
                if "language" not in row_dict:
                    row_dict["language"] = config_name
                all_rows.append(row_dict)
        except Exception as e:
            logger.warning(f"  Skipping config '{config_name}': {e}")

    logger.info(f"  Loaded {len(all_rows)} total rows across all configs")
    return all_rows


def _pull_generated_dataset(name: str, catalog_entry: dict, generator_path: str) -> None:
    """Generate a procedural dataset and register it locally.

    The generator function is specified as 'module:function' and must return
    a list of dicts for each split. It receives (split, catalog_entry) as args.
    """
    from rllm.data import DatasetRegistry

    gen_fn = _load_transform(generator_path)
    splits = catalog_entry.get("splits", ["train", "test"])

    for split in splits:
        try:
            data_list = gen_fn(split, catalog_entry)
            DatasetRegistry.register_dataset(
                name=name,
                data=data_list,
                split=split,
                source=catalog_entry.get("source", "generated"),
                description=catalog_entry.get("description", ""),
                category=catalog_entry.get("category", ""),
            )
            logger.info(f"  Generated and registered {name}/{split} ({len(data_list)} examples)")
        except Exception as e:
            logger.error(f"  Failed to generate {name}/{split}: {e}")
            raise


def pull_dataset(name: str, catalog_entry: dict) -> None:
    """Download a dataset from HuggingFace (or generate it) and register locally.

    Supports optional field_map, hf_config, aggregate_configs, transform,
    and generator for procedurally-generated datasets.

    Args:
        name: Dataset name (e.g., 'gsm8k').
        catalog_entry: Entry from datasets.json with 'source', 'splits', etc.
    """
    # Check for a generator function (procedural datasets that don't live on HF)
    generator_path = catalog_entry.get("generator")
    if generator_path:
        _pull_generated_dataset(name, catalog_entry, generator_path)
        return

    from datasets import load_dataset

    from rllm.data import DatasetRegistry

    source = catalog_entry["source"]
    splits = catalog_entry.get("splits", ["train", "test"])
    hf_config = catalog_entry.get("hf_config")
    field_map = catalog_entry.get("field_map")
    transform_path = catalog_entry.get("transform")
    aggregate_configs = catalog_entry.get("aggregate_configs", False)

    # Load transform function if specified
    transform_fn = _load_transform(transform_path) if transform_path else None

    logger.info(f"Pulling dataset '{name}' from {source}...")

    data_files = catalog_entry.get("data_files")
    hf_split = catalog_entry.get("hf_split")

    for split in splits:
        try:
            if aggregate_configs:
                # Load all HF configs and merge into a single dataset
                # (_load_all_configs already applies _disable_image_decoding)
                data_list = _load_all_configs(source, split, hf_split)
            else:
                # Build load_dataset kwargs
                # hf_split overrides the split used in load_dataset() (e.g. when
                # data_files forces the HuggingFace split to "train" but we want to
                # register it under "test").
                load_kwargs: dict = {"path": source, "split": hf_split or split}
                if hf_config:
                    load_kwargs["name"] = hf_config
                if data_files:
                    load_kwargs["data_files"] = data_files

                hf_dataset = load_dataset(**load_kwargs)
                hf_dataset = _disable_image_decoding(hf_dataset)
                data_list = None

            # Convert to list of dicts for transformation
            if data_list is not None:
                # Already a list (from aggregate_configs)
                if transform_fn:
                    data_list = [r for row in data_list if (r := transform_fn(row)) is not None]

                if field_map:
                    data_list = [_remap_fields(row, field_map) for row in data_list]

                # Flatten image dicts to raw bytes
                data_list = _flatten_image_dicts(data_list)

                register_data = data_list
            elif transform_fn or field_map:
                data_list = [dict(row) for row in hf_dataset]

                if transform_fn:
                    data_list = [r for row in data_list if (r := transform_fn(row)) is not None]

                if field_map:
                    data_list = [_remap_fields(row, field_map) for row in data_list]

                # Flatten image dicts to raw bytes
                data_list = _flatten_image_dicts(data_list)

                register_data = data_list
            else:
                # Even without transforms, flatten any image dicts
                data_list = [dict(row) for row in hf_dataset]
                data_list = _flatten_image_dicts(data_list)
                register_data = data_list

            DatasetRegistry.register_dataset(
                name=name,
                data=register_data,
                split=split,
                source=source,
                description=catalog_entry.get("description", ""),
                category=catalog_entry.get("category", ""),
            )
            num_examples = len(register_data)
            logger.info(f"  Registered {name}/{split} ({num_examples} examples)")
        except Exception as e:
            # Provide a clearer message for gated datasets
            err_str = str(e)
            if "gated dataset" in err_str.lower() or "must be authenticated" in err_str.lower():
                logger.warning(f"  Failed to pull {name}/{split}: This is a gated dataset. Set the HF_TOKEN environment variable or run 'huggingface-cli login' to authenticate.")
            else:
                logger.warning(f"  Failed to pull {name}/{split}: {e}")
            raise
