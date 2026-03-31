import json
import logging
import os
import shutil
from typing import Any

logger = logging.getLogger(__name__)


class Dataset:
    """A class representing a dataset.

    Implements the ``__len__``/``__getitem__`` protocol expected by
    ``torch.utils.data.DataLoader`` without requiring torch at import time.
    """

    def __init__(self, data: list[dict[str, Any]], name: str | None = None, split: str | None = None):
        """Initialize a Dataset.

        Args:
            data: List of dictionaries containing the dataset examples
            name: Optional name for the dataset
            split: Optional split name (e.g., 'train', 'test')
        """
        self.data = data
        self.name = name
        self.split = split

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get an item by index."""
        return self.data[idx]

    def get_data(self) -> list[dict[str, Any]]:
        """Get the dataset data."""
        return self.data

    def repeat(self, n: int) -> "Dataset":
        """Repeat the dataset n times, keeping repeated entries adjacent.

        Args:
            n: Number of times to repeat the dataset

        Returns:
            Dataset: A new dataset with repeated entries
        """
        if n <= 0:
            raise ValueError("Repeat count must be positive")

        # Create repeated data with adjacent copies
        repeated_data = []
        for item in self.data:
            # Add n copies of this item consecutively
            repeated_data.extend([item.copy() for _ in range(n)])

        return Dataset(data=repeated_data, name=self.name, split=self.split)

    def shuffle(self, seed: int | None = None) -> "Dataset":
        """Shuffle the dataset (HuggingFace-compatible API).

        Args:
            seed: Random seed for reproducibility

        Returns:
            Dataset: A new shuffled dataset
        """
        import random

        indices = list(range(len(self.data)))
        if seed is not None:
            random.Random(seed).shuffle(indices)
        else:
            random.shuffle(indices)

        shuffled_data = [self.data[i] for i in indices]
        return Dataset(data=shuffled_data, name=self.name, split=self.split)

    def select(self, indices: list[int] | range) -> "Dataset":
        """Select a subset of the dataset (HuggingFace-compatible API).

        Args:
            indices: List or range of indices to select

        Returns:
            Dataset: A new dataset with selected examples
        """
        if isinstance(indices, range):
            indices = list(indices)

        selected_data = [self.data[i] for i in indices if 0 <= i < len(self.data)]
        return Dataset(data=selected_data, name=self.name, split=self.split)

    def get_data_path(self) -> str | None:
        """Get the absolute path of the dataset file.

        Returns:
            Optional[str]: The absolute path of the dataset file, or None if the dataset is not registered
        """
        if self.name is None or self.split is None:
            return None

        registry = DatasetRegistry._load_registry()
        datasets = registry.get("datasets", {})
        if self.name not in datasets:
            return None
        splits = datasets[self.name].get("splits", {})
        if self.split not in splits:
            return None

        return DatasetRegistry._resolve_path(splits[self.split]["path"])

    def get_verl_data_path(self) -> str | None:
        """Get the absolute path of the Verl-processed dataset file.

        Returns:
            Optional[str]: The absolute path of the Verl-processed dataset file, or None if not found
        """
        data_path = self.get_data_path()
        if data_path is None:
            return None

        # .arrow files store the verl companion as _verl.parquet
        if data_path.endswith(".arrow"):
            verl_path = data_path[: -len(".arrow")] + "_verl.parquet"
        else:
            verl_path = data_path.replace(".parquet", "_verl.parquet")
        return verl_path if os.path.exists(verl_path) else None

    @classmethod
    def load_data(cls, path: str) -> "Dataset":
        """Load dataset directly from a file path.

        Args:
            path: Path to the dataset file

        Returns:
            Dataset: The loaded dataset

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")

        file_ext = os.path.splitext(path)[1].lower()

        if file_ext == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        elif file_ext == ".jsonl":
            data = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif file_ext == ".csv":
            import pandas as pd

            data = pd.read_csv(path).to_dict("records")
        elif file_ext == ".parquet":
            import pandas as pd

            data = pd.read_parquet(path).to_dict("records")
        elif file_ext == ".arrow":
            data = DatasetRegistry._load_arrow_ipc(path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return cls(data=data)


class DatasetRegistry:
    """A registry for datasets that manages storage and retrieval.

    Uses ~/.rllm/datasets/ for storage with a v2 JSON registry format
    that includes metadata alongside file paths.
    """

    _RLLM_HOME = os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))
    _REGISTRY_FILE = os.path.join(_RLLM_HOME, "datasets", "registry.json")
    _DATASET_DIR = os.path.join(_RLLM_HOME, "datasets")

    # Legacy paths for v1 migration
    _LEGACY_REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "registry")
    _LEGACY_REGISTRY_FILE = os.path.join(_LEGACY_REGISTRY_DIR, "dataset_registry.json")
    _LEGACY_DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")

    @classmethod
    def _ensure_directories(cls) -> None:
        """Ensure the registry and dataset directories exist."""
        os.makedirs(cls._DATASET_DIR, exist_ok=True)

    @classmethod
    def _migrate_v1_to_v2(cls, v1_data: dict) -> dict:
        """Migrate a v1 flat registry to v2 format.

        v1 format: { "name": { "split": "/abs/path.parquet" } }
        v2 format: { "version": 2, "datasets": { "name": { "metadata": {...}, "splits": { "split": { "path": "rel/path.parquet", ... } } } } }
        """
        v2: dict[str, Any] = {"version": 2, "datasets": {}}
        for name, splits in v1_data.items():
            v2_entry: dict[str, Any] = {"metadata": {}, "splits": {}}
            for split, abs_path in splits.items():
                # Copy file to new location if it exists at old path
                rel_path = os.path.join(name, f"{split}.parquet")
                new_abs_path = os.path.join(cls._DATASET_DIR, rel_path)
                if abs_path and os.path.exists(abs_path) and not os.path.exists(new_abs_path):
                    os.makedirs(os.path.dirname(new_abs_path), exist_ok=True)
                    try:
                        os.symlink(os.path.abspath(abs_path), new_abs_path)
                    except OSError:
                        import shutil

                        shutil.copy2(abs_path, new_abs_path)
                    # Also handle verl files
                    verl_old = abs_path.replace(".parquet", "_verl.parquet")
                    verl_new = new_abs_path.replace(".parquet", "_verl.parquet")
                    if os.path.exists(verl_old) and not os.path.exists(verl_new):
                        try:
                            os.symlink(os.path.abspath(verl_old), verl_new)
                        except OSError:
                            import shutil

                            shutil.copy2(verl_old, verl_new)

                split_info: dict[str, Any] = {"path": rel_path}
                # Try to get num_examples
                target = new_abs_path if os.path.exists(new_abs_path) else abs_path
                if target and os.path.exists(target):
                    try:
                        import polars as pl

                        num = len(pl.read_parquet(target))
                        split_info["num_examples"] = num
                        split_info["fields"] = list(pl.read_parquet(target).columns)
                    except Exception:
                        pass
                v2_entry["splits"][split] = split_info
            v2["datasets"][name] = v2_entry

        logger.info("Migrated dataset registry from v1 to v2 format.")
        return v2

    @classmethod
    def _load_registry(cls) -> dict:
        """Load the dataset registry, auto-migrating from v1 if needed.

        Returns v2-format dict. For backward compat, callers that used the old
        flat dict can still use the public methods unchanged.
        """
        cls._ensure_directories()

        # Try loading from new location first
        if os.path.exists(cls._REGISTRY_FILE):
            try:
                with open(cls._REGISTRY_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("version") == 2:
                    return data
                # Old format at new location — migrate in place
                v2 = cls._migrate_v1_to_v2(data)
                cls._save_registry(v2)
                return v2
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in registry file. Creating a new registry.")
                return {"version": 2, "datasets": {}}

        # Try migrating from legacy location
        if os.path.exists(cls._LEGACY_REGISTRY_FILE):
            try:
                with open(cls._LEGACY_REGISTRY_FILE, encoding="utf-8") as f:
                    v1_data = json.load(f)
                v2 = cls._migrate_v1_to_v2(v1_data)
                cls._save_registry(v2)
                return v2
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to migrate legacy registry: {e}")

        return {"version": 2, "datasets": {}}

    @classmethod
    def _save_registry(cls, registry: dict) -> None:
        """Save the dataset registry to the registry file."""
        cls._ensure_directories()
        with open(cls._REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    @classmethod
    def _resolve_path(cls, rel_path: str) -> str:
        """Resolve a relative dataset path to an absolute path."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(cls._DATASET_DIR, rel_path)

    @classmethod
    def _has_binary_columns(cls, data_list: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        """Detect columns containing ``bytes`` or ``list[bytes]`` in the first row.

        Returns:
            Tuple of (has_binary, list_of_binary_column_names).
        """
        if not data_list:
            return False, []

        first = data_list[0]
        bin_cols: list[str] = []
        for col, val in first.items():
            if isinstance(val, bytes):
                bin_cols.append(col)
            elif isinstance(val, list) and val and isinstance(val[0], bytes):
                bin_cols.append(col)
        return bool(bin_cols), bin_cols

    @classmethod
    def _save_arrow_ipc(cls, data_list: list[dict[str, Any]], path: str) -> None:
        """Save data as an Arrow IPC (Feather v2) file.

        PyArrow natively handles ``bytes`` and ``list<bytes>`` columns.

        Args:
            data_list: List of row dicts.
            path: Absolute path to write the ``.arrow`` file.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        # Build column arrays from data_list
        if not data_list:
            table = pa.table({})
        else:
            columns: dict[str, list] = {k: [] for k in data_list[0]}
            for row in data_list:
                for k in columns:
                    columns[k].append(row.get(k))
            table = pa.table(columns)

        with pa.OSFile(path, "wb") as f:
            writer = ipc.new_file(f, table.schema)
            writer.write_table(table)
            writer.close()

    @classmethod
    def _load_arrow_ipc(cls, path: str) -> list[dict[str, Any]]:
        """Load data from an Arrow IPC file.

        Args:
            path: Absolute path to the ``.arrow`` file.

        Returns:
            List of row dicts. Binary columns are returned as ``bytes``.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with pa.OSFile(path, "rb") as f:
            reader = ipc.open_file(f)
            table = reader.read_all()

        col_dict = {name: table.column(name).to_pylist() for name in table.column_names}
        num_rows = table.num_rows
        return [{col: col_dict[col][i] for col in col_dict} for i in range(num_rows)]

    @classmethod
    def _strip_binary_columns(cls, data_list: list[dict[str, Any]], bin_cols: list[str]) -> list[dict[str, Any]]:
        """Return copies of rows with binary columns set to ``None``.

        Used for verl postprocessing where image data isn't needed.
        """
        stripped = []
        for row in data_list:
            new_row = dict(row)
            for col in bin_cols:
                new_row[col] = None
            stripped.append(new_row)
        return stripped

    @classmethod
    def _verl_path_for(cls, dataset_path: str) -> str:
        """Derive the verl companion parquet path from any dataset path."""
        if dataset_path.endswith(".arrow"):
            return dataset_path[: -len(".arrow")] + "_verl.parquet"
        return dataset_path.replace(".parquet", "_verl.parquet")

    @classmethod
    def register_dataset(cls, name: str, data: list[dict[str, Any]] | Any, split: str = "default", source: str = "", description: str = "", category: str = "") -> Dataset:
        """Register a dataset by saving it to disk and updating the registry.

        Args:
            name: Name of the dataset
            data: List of dictionaries containing the dataset examples or a Hugging Face dataset
            split: Split name (e.g., 'train', 'test', 'default')
            source: Optional source identifier (e.g., HuggingFace repo)
            description: Optional description
            category: Optional category (e.g., 'math', 'code')

        Returns:
            Dataset: The registered dataset
        """
        cls._ensure_directories()

        # Create dataset directory if it doesn't exist
        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        import pandas as pd

        # Convert HuggingFace dataset to list of dictionaries if needed
        if hasattr(data, "to_pandas") and callable(data.to_pandas):
            # This is likely a HuggingFace dataset
            data_df = data.to_pandas()
            data_list = data_df.to_dict("records")
        else:
            # Assume it's already a list of dictionaries
            data_list = data

        # Detect binary columns (image bytes)
        has_binary, bin_cols = cls._has_binary_columns(data_list)

        if has_binary:
            # Save as Arrow IPC (handles binary columns natively)
            rel_path = os.path.join(name, f"{split}.arrow")
            dataset_path = os.path.join(cls._DATASET_DIR, rel_path)
            cls._save_arrow_ipc(data_list, dataset_path)

            # Strip binary columns for verl postprocessing (images not needed)
            stripped = cls._strip_binary_columns(data_list, bin_cols)
            verl_data = cls.apply_verl_postprocessing(stripped)
            verl_dataset_path = cls._verl_path_for(dataset_path)
            verl_data_df = pd.DataFrame(verl_data)
            verl_data_df.to_parquet(verl_dataset_path)

            fields = list(data_list[0].keys()) if data_list else []
        else:
            # Text-only: use parquet (existing path)
            data_df = pd.DataFrame(data_list)
            rel_path = os.path.join(name, f"{split}.parquet")
            dataset_path = os.path.join(cls._DATASET_DIR, rel_path)
            data_df.to_parquet(dataset_path)

            # Apply Verl postprocessing and save
            verl_data = cls.apply_verl_postprocessing(data_list)
            verl_dataset_path = cls._verl_path_for(dataset_path)
            verl_data_df = pd.DataFrame(verl_data)
            verl_data_df.to_parquet(verl_dataset_path)

            fields = list(data_df.columns)

        # Update registry (v2 format)
        registry = cls._load_registry()
        datasets = registry.setdefault("datasets", {})

        if name not in datasets:
            datasets[name] = {"metadata": {}, "splits": {}}

        # Update metadata if provided
        entry = datasets[name]
        if source:
            entry["metadata"]["source"] = source
        if description:
            entry["metadata"]["description"] = description
        if category:
            entry["metadata"]["category"] = category

        # Record field names and count
        entry["splits"][split] = {"path": rel_path, "num_examples": len(data_list), "fields": fields}
        cls._save_registry(registry)

        logger.info(f"Registered dataset '{name}' split '{split}' with {len(data_list)} examples.")

        return Dataset(data=data_list, name=name, split=split)

    @classmethod
    def load_dataset(cls, name: str, split: str = "default") -> Dataset | None:
        """Load a dataset from the registry.

        Args:
            name: Name of the dataset to load
            split: Split name to load (e.g., 'train', 'test', 'default')

        Returns:
            Dataset: The loaded dataset or None if not found
        """
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})

        if name not in datasets:
            logger.warning(f"Dataset '{name}' not found in registry.")
            return None

        splits = datasets[name].get("splits", {})
        if split not in splits:
            logger.warning(f"Split '{split}' not found in dataset '{name}'.")
            return None

        # Load data — resolve relative path
        split_info = splits[split]
        dataset_path = cls._resolve_path(split_info["path"])
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return None

        if dataset_path.endswith(".arrow"):
            data = cls._load_arrow_ipc(dataset_path)
        else:
            import polars as pl

            data = pl.read_parquet(dataset_path).to_dicts()

        logger.info(f"Loaded dataset '{name}' split '{split}' with {len(data)} examples.")

        return Dataset(data=data, name=name, split=split)

    @classmethod
    def get_dataset_names(cls) -> list[str]:
        """Get the names of all registered datasets.

        Returns:
            List[str]: List of dataset names
        """
        registry = cls._load_registry()
        return list(registry.get("datasets", {}).keys())

    @classmethod
    def get_dataset_splits(cls, name: str) -> list[str]:
        """Get the available splits for a dataset.

        Args:
            name: Name of the dataset

        Returns:
            List[str]: List of available splits
        """
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})
        if name not in datasets:
            return []
        return list(datasets[name].get("splits", {}).keys())

    @classmethod
    def dataset_exists(cls, name: str, split: str | None = None) -> bool:
        """Check if a dataset exists in the registry.

        Args:
            name: Name of the dataset to check
            split: Optional split to check

        Returns:
            bool: True if the dataset exists, False otherwise
        """
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})
        if name not in datasets:
            return False
        if split is not None:
            return split in datasets[name].get("splits", {})
        return True

    @classmethod
    def get_dataset_info(cls, name: str) -> dict | None:
        """Get metadata and split info for a dataset.

        Args:
            name: Name of the dataset

        Returns:
            dict with 'metadata' and 'splits' keys, or None if not found
        """
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})
        return datasets.get(name)

    @classmethod
    def remove_dataset_split(cls, name: str, split: str) -> bool:
        """Remove a specific split from a dataset in the registry.

        Args:
            name: Name of the dataset
            split: Split to remove

        Returns:
            bool: True if the split was removed, False otherwise
        """
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})
        if name not in datasets or split not in datasets[name].get("splits", {}):
            logger.warning(f"Dataset '{name}' split '{split}' not found in registry.")
            return False

        # Get dataset path
        split_info = datasets[name]["splits"][split]
        dataset_path = cls._resolve_path(split_info["path"])

        # Remove file if it exists
        if dataset_path and os.path.exists(dataset_path):
            os.remove(dataset_path)

        # Also remove the Verl-processed file if it exists
        verl_path = cls._verl_path_for(dataset_path)
        if os.path.exists(verl_path):
            os.remove(verl_path)

        # Remove split from registry
        del datasets[name]["splits"][split]

        # If no splits left, remove the dataset entry and clean up
        if not datasets[name]["splits"]:
            del datasets[name]
            dataset_dir = os.path.join(cls._DATASET_DIR, name)
            # Clean up legacy images directory
            images_dir = os.path.join(dataset_dir, "images")
            if os.path.isdir(images_dir):
                shutil.rmtree(images_dir)
            if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
                os.rmdir(dataset_dir)

        cls._save_registry(registry)

        logger.info(f"Removed dataset '{name}' split '{split}' from registry.")
        return True

    @classmethod
    def remove_dataset(cls, name: str) -> bool:
        """Remove an entire dataset from the registry and delete its files.

        Args:
            name: Name of the dataset to remove

        Returns:
            bool: True if the dataset was removed, False otherwise
        """
        registry = cls._load_registry()
        datasets = registry.get("datasets", {})
        if name not in datasets:
            logger.warning(f"Dataset '{name}' not found in registry.")
            return False

        # Remove files for all splits
        for split, split_info in datasets[name].get("splits", {}).items():
            path = cls._resolve_path(split_info["path"])
            if path and os.path.exists(path):
                os.remove(path)
            verl_path = cls._verl_path_for(path)
            if os.path.exists(verl_path):
                os.remove(verl_path)

        # Clean up legacy images directory and dataset directory
        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        images_dir = os.path.join(dataset_dir, "images")
        if os.path.isdir(images_dir):
            shutil.rmtree(images_dir)
        if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
            os.rmdir(dataset_dir)

        # Update registry
        del datasets[name]
        cls._save_registry(registry)

        logger.info(f"Removed dataset '{name}' from registry.")
        return True

    @classmethod
    def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply Verl postprocessing to the dataset.

        Args:
            data: List of dictionaries containing the dataset examples

        Returns:
            List of dictionaries with Verl-compatible format
        """
        processed_data = []
        for entry in data:
            processed_entry = {
                "prompt": [{"role": "user", "content": "placeholder"}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None,
                },
                "extra_info": entry,
            }
            processed_data.append(processed_entry)
        return processed_data
