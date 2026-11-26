from __future__ import annotations

import json

import pytest

from rllm.tasks.loader import BenchmarkLoader, BenchmarkSplitError


def test_vlm_instruction_encodes_images_lazily(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    images = bench / "images"
    data = bench / "data"
    images.mkdir(parents=True)
    data.mkdir()

    (images / "sample.png").write_bytes(b"not-a-real-png")
    (data / "test.jsonl").write_text('{"id":"0","question":"what?","image":"images/sample.png"}\n')
    (bench / "dataset.toml").write_text('[dataset]\nname = "bench"\ncategory = "vlm"\ninstruction_field = "question"\n')

    calls: list[tuple[str, str]] = []

    def fake_data_uri(path: str, mime: str) -> str:
        calls.append((path, mime))
        return f"data:{mime};base64,ZmFrZQ=="

    monkeypatch.setattr("rllm.tasks.loader._image_file_to_data_uri", fake_data_uri)

    result = BenchmarkLoader.load(str(bench))
    instruction = result.tasks[0].instruction

    assert isinstance(instruction, list)
    assert calls == []

    payload = json.loads(json.dumps(instruction))

    assert calls == [(str((images / "sample.png").resolve()), "image/png")]
    assert payload == [
        {"type": "text", "text": "what?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
    ]


@pytest.mark.parametrize("layout", ["fixed_file", "task_directories"])
def test_load_rejects_split_conflicting_with_fixed_layout(tmp_path, layout):
    config = '[dataset]\nsplit = "train"\n'
    if layout == "fixed_file":
        config += 'data = "questions.jsonl"\n'
        (tmp_path / "questions.jsonl").write_text('{"question":"math question"}\n')
        expected_instruction = "math question"
    else:
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text("[task]\n")
        (task / "instruction.md").write_text("task instruction")
        expected_instruction = "task instruction"
    (tmp_path / "dataset.toml").write_text(config)

    with pytest.raises(BenchmarkSplitError, match="only provides split 'train', but split 'test' was requested"):
        BenchmarkLoader.load(str(tmp_path), split="test")

    result = BenchmarkLoader.load(str(tmp_path), split="train")
    assert result.split == "train"
    assert [task.instruction for task in result.tasks] == [expected_instruction]


def test_load_only_falls_back_when_split_is_omitted(tmp_path):
    (tmp_path / "dataset.toml").write_text('[dataset]\nname = "legacy"\n')
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"question":"train question"}\n')

    with pytest.raises(BenchmarkSplitError, match=r"Split 'test' is missing its data file: .*/data/test\.jsonl"):
        BenchmarkLoader.load(str(tmp_path), split="test")

    result = BenchmarkLoader.load(str(tmp_path))

    assert result.split == "train"
    assert [task.instruction for task in result.tasks] == ["train question"]
