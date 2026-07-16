"""Unit tests for MultimodalCodexHarness — image-bytes upload behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

from rllm.types import AgentConfig, Task

from cookbooks.multimodal_codex.harness import MultimodalCodexHarness


@dataclass
class _UploadCall:
    local_path: str
    remote_path: str
    local_bytes: bytes


@dataclass
class _ExecCall:
    command: str
    user: str | None
    timeout: float | None


@dataclass
class FakeSandbox:
    """Records exec() and upload_file() calls."""

    stdout: str = "OK"
    exec_calls: list[_ExecCall] = field(default_factory=list)
    upload_calls: list[_UploadCall] = field(default_factory=list)

    def exec(self, command: str, timeout: float | None = None, user: str | None = None) -> str:
        self.exec_calls.append(_ExecCall(command=command, user=user, timeout=timeout))
        return self.stdout

    def upload_file(self, local_path: str, remote_path: str) -> None:
        with open(local_path, "rb") as f:
            blob = f.read()
        self.upload_calls.append(_UploadCall(local_path=local_path, remote_path=remote_path, local_bytes=blob))

    def upload_dir(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


PNG_SIG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PNG_2 = b"\x89PNG\r\n\x1a\n" + b"\xff" * 32


def _config() -> AgentConfig:
    return AgentConfig(base_url="http://gw:8000/sessions/s/v1", model="openai/gpt-4o", session_uid="s")


def test_write_configs_uploads_single_image():
    task = Task(
        id="t1",
        instruction="q",
        metadata={"image_bytes": PNG_SIG, "image_file": "/tmp/multimodal_codex_input.png"},
    )
    sandbox = FakeSandbox()
    h = MultimodalCodexHarness()

    h.write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk-fake"})

    assert len(sandbox.upload_calls) == 1
    call = sandbox.upload_calls[0]
    assert call.remote_path == "/tmp/multimodal_codex_input.png"
    assert call.local_bytes == PNG_SIG


def test_write_configs_uploads_multi_images():
    task = Task(
        id="t1",
        instruction="q",
        metadata={
            "image_files": ["/tmp/a.png", "/tmp/b.png"],
            "images_bytes": [PNG_SIG, PNG_2],
        },
    )
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})

    assert [c.remote_path for c in sandbox.upload_calls] == ["/tmp/a.png", "/tmp/b.png"]
    assert [c.local_bytes for c in sandbox.upload_calls] == [PNG_SIG, PNG_2]


def test_write_configs_skips_when_no_image():
    task = Task(id="t1", instruction="q", metadata={"ground_truth": "42"})
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})

    assert sandbox.upload_calls == []


def test_write_configs_still_writes_codex_auth_and_toml():
    """The subclass MUST call super() — auth.json + config.toml still land."""
    task = Task(
        id="t1",
        instruction="q",
        metadata={"image_bytes": PNG_SIG, "image_file": "/tmp/img.png"},
    )
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})

    exec_cmd = " ".join(c.command for c in sandbox.exec_calls)
    assert "auth.json" in exec_cmd
    assert "config.toml" in exec_cmd


def test_local_tmp_file_cleaned_up_after_upload():
    import os as _os

    task = Task(
        id="t1",
        instruction="q",
        metadata={"image_bytes": PNG_SIG, "image_file": "/tmp/img.png"},
    )
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})

    local_path = sandbox.upload_calls[0].local_path
    assert not _os.path.exists(local_path), f"tmp file leaked: {local_path}"


def test_write_configs_no_metadata_is_noop():
    task = Task(id="t1", instruction="q", metadata=None)
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})
    # Base class still writes auth files, but no uploads happen.
    assert sandbox.upload_calls == []


def test_multi_shape_zero_length_lists():
    task = Task(
        id="t1",
        instruction="q",
        metadata={"image_files": [], "images_bytes": []},
    )
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})
    assert sandbox.upload_calls == []


def test_upload_handles_verl_wrapped_dict():
    """verl parquet path stores image_bytes as {'bytes':..., 'path':''} —
    DatasetRegistry._wrap_binary_columns_for_parquet wraps raw ``bytes``
    columns into that shape for verl's rl_dataset._build_messages. Harness
    must unwrap before writing to tmp file (else f.write(dict) → TypeError)."""
    task = Task(
        id="t1",
        instruction="q",
        metadata={
            "image_bytes": {"bytes": PNG_SIG, "path": ""},
            "image_file": "/tmp/multimodal_codex_input.png",
        },
    )
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})

    assert len(sandbox.upload_calls) == 1
    call = sandbox.upload_calls[0]
    assert call.remote_path == "/tmp/multimodal_codex_input.png"
    assert call.local_bytes == PNG_SIG


def test_upload_handles_verl_wrapped_list():
    """Same wrapping applies element-wise for list columns."""
    task = Task(
        id="t1",
        instruction="q",
        metadata={
            "image_files": ["/tmp/a.png", "/tmp/b.png"],
            "images_bytes": [
                {"bytes": PNG_SIG, "path": ""},
                {"bytes": PNG_2, "path": ""},
            ],
        },
    )
    sandbox = FakeSandbox()
    MultimodalCodexHarness().write_configs(sandbox, task, _config(), env={"OPENAI_API_KEY": "sk"})

    assert [c.remote_path for c in sandbox.upload_calls] == ["/tmp/a.png", "/tmp/b.png"]
    assert [c.local_bytes for c in sandbox.upload_calls] == [PNG_SIG, PNG_2]
