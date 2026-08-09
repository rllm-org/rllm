from __future__ import annotations

import json

from rllm.tasks.loader import BenchmarkLoader


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


def test_harbor_verifier_env_is_lifted_into_task_metadata(tmp_path):
    bench = tmp_path / "bench"
    task_dir = bench / "task-1"
    task_dir.mkdir(parents=True)
    (bench / "dataset.toml").write_text('[dataset]\nname = "bench"\ntype = "sandbox"\n')
    (task_dir / "instruction.md").write_text("do work")
    (task_dir / "task.toml").write_text(
        '[task]\nname = "task-1"\n\n[verifier]\nscript = "tests/test.sh"\n\n[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
    )

    result = BenchmarkLoader.load(str(bench))

    assert result.tasks[0].metadata["verifier_env"] == {"OPENAI_API_KEY": "${OPENAI_API_KEY}"}
