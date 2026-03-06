"""Tests for dataset loading."""

import json

import pytest
from rllm_telemetry.dataset import _apply_field_map, _load_file, load_dataset

# ---------------------------------------------------------------------------
# _apply_field_map
# ---------------------------------------------------------------------------


class TestApplyFieldMap:
    def test_default_rllm_mapping(self):
        row = {"question": "What is 2+2?", "ground_truth": "4", "data_source": "math"}
        mapped = _apply_field_map(row, {"question": "input", "ground_truth": "expected"})
        assert mapped["input"] == "What is 2+2?"
        assert mapped["expected"] == "4"
        assert mapped["data_source"] == "math"  # unmapped fields preserved

    def test_custom_mapping(self):
        row = {"prompt": "Say hi", "answer": "Hello"}
        mapped = _apply_field_map(row, {"prompt": "input", "answer": "expected"})
        assert mapped["input"] == "Say hi"
        assert mapped["expected"] == "Hello"

    def test_already_correct_fields(self):
        row = {"input": "hello", "expected": "world"}
        mapped = _apply_field_map(row, {"question": "input", "ground_truth": "expected"})
        assert mapped["input"] == "hello"
        assert mapped["expected"] == "world"


# ---------------------------------------------------------------------------
# _load_file
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_json_array(self, tmp_path):
        path = tmp_path / "data.json"
        data = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        path.write_text(json.dumps(data))
        result = _load_file(str(path))
        assert len(result) == 2
        assert result[0]["question"] == "Q1"

    def test_json_wrapped(self, tmp_path):
        path = tmp_path / "data.json"
        data = {"data": [{"q": "Q1"}, {"q": "Q2"}]}
        path.write_text(json.dumps(data))
        result = _load_file(str(path))
        assert len(result) == 2

    def test_json_wrapped_examples_key(self, tmp_path):
        path = tmp_path / "data.json"
        data = {"examples": [{"q": "Q1"}]}
        path.write_text(json.dumps(data))
        result = _load_file(str(path))
        assert len(result) == 1

    def test_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        lines = [json.dumps({"q": f"Q{i}"}) for i in range(3)]
        path.write_text("\n".join(lines))
        result = _load_file(str(path))
        assert len(result) == 3

    def test_jsonl_with_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"q": "Q1"}\n\n{"q": "Q2"}\n')
        result = _load_file(str(path))
        assert len(result) == 2

    def test_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("question,answer\nQ1,A1\nQ2,A2\n")
        result = _load_file(str(path))
        assert len(result) == 2
        assert result[0]["question"] == "Q1"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _load_file("/nonexistent/file.json")

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            _load_file(str(path))


# ---------------------------------------------------------------------------
# load_dataset (integration)
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_from_jsonl_with_default_mapping(self, tmp_path):
        path = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"question": "What is 2+2?", "ground_truth": "4"}),
            json.dumps({"question": "Capital of France?", "ground_truth": "Paris"}),
        ]
        path.write_text("\n".join(lines))

        result = load_dataset(str(path))
        assert len(result) == 2
        assert result[0]["input"] == "What is 2+2?"
        assert result[0]["expected"] == "4"

    def test_custom_field_map(self, tmp_path):
        path = tmp_path / "data.json"
        data = [{"prompt": "Hi", "response": "Hello"}]
        path.write_text(json.dumps(data))

        result = load_dataset(str(path), field_map={"prompt": "input", "response": "expected"})
        assert result[0]["input"] == "Hi"
        assert result[0]["expected"] == "Hello"

    def test_limit(self, tmp_path):
        path = tmp_path / "data.json"
        data = [{"question": f"Q{i}", "ground_truth": f"A{i}"} for i in range(10)]
        path.write_text(json.dumps(data))

        result = load_dataset(str(path), limit=3)
        assert len(result) == 3

    def test_nonexistent_registry_name(self):
        with pytest.raises((ImportError, ValueError)):
            load_dataset("nonexistent_dataset_xyz_123")
