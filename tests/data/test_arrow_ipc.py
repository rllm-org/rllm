"""Tests for Arrow IPC save/load and format-aware dataset registry."""

import os

import pytest

from rllm.data.dataset import Dataset, DatasetRegistry


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Set up a temporary RLLM_HOME directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", rllm_home)
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", os.path.join(rllm_home, "datasets", "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", os.path.join(rllm_home, "datasets"))
    return rllm_home


# ---------------------------------------------------------------------------
# Arrow IPC roundtrip
# ---------------------------------------------------------------------------


class TestArrowIPCRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        """Binary columns survive Arrow IPC save → load cycle."""
        data = [
            {"text": "hello", "img": b"\x89PNG\x00\x01\x02"},
            {"text": "world", "img": b"\xff\xd8\xff\x03\x04"},
        ]
        path = str(tmp_path / "test.arrow")
        DatasetRegistry._save_arrow_ipc(data, path)
        loaded = DatasetRegistry._load_arrow_ipc(path)

        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"
        assert loaded[0]["img"] == b"\x89PNG\x00\x01\x02"
        assert loaded[1]["text"] == "world"
        assert loaded[1]["img"] == b"\xff\xd8\xff\x03\x04"

    def test_list_binary_column(self, tmp_path):
        """list<binary> columns survive Arrow IPC roundtrip."""
        data = [
            {"text": "q", "images": [b"img1", b"img2"]},
            {"text": "r", "images": [b"img3"]},
        ]
        path = str(tmp_path / "test.arrow")
        DatasetRegistry._save_arrow_ipc(data, path)
        loaded = DatasetRegistry._load_arrow_ipc(path)

        assert loaded[0]["images"] == [b"img1", b"img2"]
        assert loaded[1]["images"] == [b"img3"]

    def test_empty_data(self, tmp_path):
        """Empty data list roundtrips."""
        path = str(tmp_path / "empty.arrow")
        DatasetRegistry._save_arrow_ipc([], path)
        loaded = DatasetRegistry._load_arrow_ipc(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# _has_binary_columns
# ---------------------------------------------------------------------------


class TestHasBinaryColumns:
    def test_detects_bytes(self):
        data = [{"text": "hi", "img": b"\x89PNG"}]
        has, cols = DatasetRegistry._has_binary_columns(data)
        assert has is True
        assert cols == ["img"]

    def test_detects_list_bytes(self):
        data = [{"text": "hi", "images": [b"a", b"b"]}]
        has, cols = DatasetRegistry._has_binary_columns(data)
        assert has is True
        assert cols == ["images"]

    def test_no_binary(self):
        data = [{"text": "hi", "label": 1}]
        has, cols = DatasetRegistry._has_binary_columns(data)
        assert has is False
        assert cols == []

    def test_empty_data(self):
        has, cols = DatasetRegistry._has_binary_columns([])
        assert has is False
        assert cols == []


# ---------------------------------------------------------------------------
# register / load format dispatch
# ---------------------------------------------------------------------------


class TestRegisterFormatDispatch:
    def test_vlm_dataset_uses_arrow(self, tmp_rllm_home):
        """Data with bytes columns should be saved as .arrow."""
        data = [
            {"text": "q1", "img": b"\x89PNG_data"},
            {"text": "q2", "img": b"\xff\xd8\xff_data"},
        ]
        ds = DatasetRegistry.register_dataset("vlm_test", data, split="test")

        assert ds.name == "vlm_test"
        assert len(ds) == 2

        # Check registry path ends with .arrow
        registry = DatasetRegistry._load_registry()
        path = registry["datasets"]["vlm_test"]["splits"]["test"]["path"]
        assert path.endswith(".arrow")

        # Verify .arrow file exists
        abs_path = DatasetRegistry._resolve_path(path)
        assert os.path.exists(abs_path)

        # Verify verl companion exists
        verl_path = DatasetRegistry._verl_path_for(abs_path)
        assert os.path.exists(verl_path)

    def test_text_dataset_uses_parquet(self, tmp_rllm_home):
        """Data without binary columns should use .parquet."""
        data = [{"question": "hi", "answer": "42"}]
        DatasetRegistry.register_dataset("text_test", data, split="test")

        registry = DatasetRegistry._load_registry()
        path = registry["datasets"]["text_test"]["splits"]["test"]["path"]
        assert path.endswith(".parquet")

    def test_load_arrow_dataset(self, tmp_rllm_home):
        """Loading an Arrow dataset should return bytes columns."""
        data = [{"text": "q", "img": b"\x89PNG_test"}]
        DatasetRegistry.register_dataset("load_arrow", data, split="test")

        loaded = DatasetRegistry.load_dataset("load_arrow", "test")
        assert loaded is not None
        assert loaded[0]["img"] == b"\x89PNG_test"
        assert loaded[0]["text"] == "q"

    def test_load_parquet_dataset(self, tmp_rllm_home):
        """Loading a parquet dataset should work as before."""
        data = [{"question": "hi", "answer": "42"}]
        DatasetRegistry.register_dataset("load_pq", data, split="test")

        loaded = DatasetRegistry.load_dataset("load_pq", "test")
        assert loaded is not None
        assert loaded[0]["question"] == "hi"


# ---------------------------------------------------------------------------
# verl path derivation
# ---------------------------------------------------------------------------


class TestVerlPathFromArrow:
    def test_verl_path_for_arrow(self):
        assert DatasetRegistry._verl_path_for("/data/ds/test.arrow") == "/data/ds/test_verl.parquet"

    def test_verl_path_for_parquet(self):
        assert DatasetRegistry._verl_path_for("/data/ds/test.parquet") == "/data/ds/test_verl.parquet"

    def test_get_verl_data_path(self, tmp_rllm_home):
        """Dataset.get_verl_data_path should work for .arrow datasets."""
        data = [{"text": "q", "img": b"\x89PNG"}]
        ds = DatasetRegistry.register_dataset("verl_arrow", data, split="test")
        verl_path = ds.get_verl_data_path()
        assert verl_path is not None
        assert verl_path.endswith("_verl.parquet")
        assert os.path.exists(verl_path)


# ---------------------------------------------------------------------------
# strip binary columns
# ---------------------------------------------------------------------------


class TestStripBinaryColumns:
    def test_strips(self):
        data = [{"text": "q", "img": b"data", "label": 1}]
        result = DatasetRegistry._strip_binary_columns(data, ["img"])
        assert result[0]["img"] is None
        assert result[0]["text"] == "q"
        assert result[0]["label"] == 1
        # Original unchanged
        assert data[0]["img"] == b"data"


# ---------------------------------------------------------------------------
# remove cleanup
# ---------------------------------------------------------------------------


class TestRemoveCleanup:
    def test_remove_cleans_arrow_and_verl(self, tmp_rllm_home):
        """Removing a dataset should clean up .arrow and _verl.parquet files."""
        data = [{"text": "q", "img": b"\x89PNG"}]
        DatasetRegistry.register_dataset("rm_test", data, split="test")

        registry = DatasetRegistry._load_registry()
        path = DatasetRegistry._resolve_path(registry["datasets"]["rm_test"]["splits"]["test"]["path"])
        verl_path = DatasetRegistry._verl_path_for(path)
        assert os.path.exists(path)
        assert os.path.exists(verl_path)

        DatasetRegistry.remove_dataset("rm_test")
        assert not os.path.exists(path)
        assert not os.path.exists(verl_path)

    def test_remove_cleans_legacy_images_dir(self, tmp_rllm_home):
        """Removing a dataset should clean up legacy images/ directory."""
        # Register a text dataset first
        data = [{"text": "q", "answer": "a"}]
        DatasetRegistry.register_dataset("legacy_img", data, split="test")

        # Manually create a legacy images directory
        images_dir = os.path.join(DatasetRegistry._DATASET_DIR, "legacy_img", "images")
        os.makedirs(images_dir, exist_ok=True)
        with open(os.path.join(images_dir, "test.png"), "wb") as f:
            f.write(b"fake")

        DatasetRegistry.remove_dataset("legacy_img")
        assert not os.path.isdir(images_dir)


# ---------------------------------------------------------------------------
# Dataset.load_data with .arrow
# ---------------------------------------------------------------------------


class TestDatasetLoadDataArrow:
    def test_load_data_arrow(self, tmp_path):
        """Dataset.load_data should handle .arrow files."""
        data = [{"text": "q", "img": b"\x89PNG_data"}]
        path = str(tmp_path / "test.arrow")
        DatasetRegistry._save_arrow_ipc(data, path)

        ds = Dataset.load_data(path)
        assert len(ds) == 1
        assert ds[0]["img"] == b"\x89PNG_data"
