"""BenchmarkLoader: load a local benchmark directory into ``list[Task]``.

After PR 2, both task-per-directory and rows-with-shared-verifier shapes
return the new :class:`rllm.types.Task` abstraction. The CLI then runs
each task through :class:`rllm.runner.Runner` — same code path for both.

Three on-disk shapes recognised:

1. ``dataset.toml`` present, ``data/<split>.jsonl`` exists → simple/data
   (rows-with-shared-verifier). Each row becomes one Task; benchmark_dir
   is the directory itself, ``sub_dir`` is ``None``.

2. ``dataset.toml`` present + sub-directories with ``task.toml`` (or
   ``[[tasks]]`` listed in ``dataset.toml``) → sandbox style. Each
   sub-directory becomes one Task with ``sub_dir`` set.

3. Single ``task.toml`` at the root, or auto-discovered sub-directories
   with ``task.toml`` (no ``dataset.toml``) → still produces Tasks.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from rllm.tasks.dataset_config import DatasetConfig, load_dataset_config
from rllm.types import Task

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """What the loader returns to the CLI.

    The Runner figures out the verifier from each Task, so we don't need
    to return an Evaluator separately. ``harness_name`` is the suggested
    harness for the AgentFlow (CLI may override with ``--agent``).
    """

    tasks: list[Task]
    name: str
    split: str = "test"
    harness_name: str | None = None
    sandbox_backend: str | None = None
    description: str = ""
    category: str = ""

    # Convenience for CLI display & legacy paths
    metadata: dict = field(default_factory=dict)


class BenchmarkLoader:
    """Detect and load local benchmark directories."""

    @staticmethod
    def is_local_benchmark(benchmark: str) -> bool:
        """Check if *benchmark* is a local path containing tasks."""
        if not os.path.exists(benchmark):
            return False
        p = Path(benchmark).resolve()
        if not p.is_dir():
            return False
        if (p / "dataset.toml").exists():
            return True
        if (p / "task.toml").exists():
            return True
        return any((d / "task.toml").exists() for d in p.iterdir() if d.is_dir())

    @staticmethod
    def load(
        benchmark_path: str,
        sandbox_backend: str | None = None,
        harness_name: str | None = None,
    ) -> BenchmarkResult:
        """Load a local benchmark directory."""
        path = Path(benchmark_path).resolve()

        if (path / "dataset.toml").exists():
            config = load_dataset_config(path / "dataset.toml")
            if config.type == "simple" or _has_data_file(path, config):
                return _load_data_dataset(path, config, sandbox_backend, harness_name)
            return _load_sandbox_dataset(path, config, sandbox_backend, harness_name)

        if (path / "task.toml").exists():
            return _load_single_task(path, sandbox_backend, harness_name)

        return _load_auto_discover(path, sandbox_backend, harness_name)


# ---------------------------------------------------------------------------
# Data dataset (rows-with-shared-verifier; gsm8k-style)
# ---------------------------------------------------------------------------


def _has_data_file(path: Path, config: DatasetConfig) -> bool:
    if config.data:
        return (path / config.data).exists()
    return (path / "data").is_dir()


def _load_data_dataset(
    path: Path,
    config: DatasetConfig,
    sandbox_backend: str | None,
    harness_name: str | None,
) -> BenchmarkResult:
    """Materialised-style dataset: jsonl rows + shared verifier."""
    data_file = _resolve_data_file(path, config)
    rows = _load_jsonl(data_file)

    instruction_field, metadata_fields = _read_dataset_meta(path)
    category = _read_dataset_category(path)
    tasks: list[Task] = []
    for idx, row in enumerate(rows):
        text = _render_instruction(path, row, instruction_field)
        if category == "vlm":
            instruction: str | list[dict] = _build_multimodal_instruction(text, row, path)
        else:
            instruction = text
        task_metadata = _build_metadata(row, metadata_fields)
        tasks.append(
            Task(
                id=str(row.get("id", idx)),
                instruction=instruction,
                metadata=task_metadata,
                benchmark_dir=path,
                sub_dir=None,
            )
        )

    return BenchmarkResult(
        tasks=tasks,
        name=config.name,
        split=config.split,
        harness_name=harness_name or config.default_agent or "simple",
        sandbox_backend=sandbox_backend,
        description=config.description,
        category=category or "custom",
    )


def _resolve_data_file(path: Path, config: DatasetConfig) -> Path:
    if config.data:
        return path / config.data
    # Default: data/<split>.jsonl
    candidate = path / "data" / f"{config.split}.jsonl"
    if candidate.exists():
        return candidate
    # Fallback: any .jsonl in data/
    data_dir = path / "data"
    if data_dir.is_dir():
        for f in sorted(data_dir.glob("*.jsonl")):
            return f
    raise FileNotFoundError(f"No data file found in {path}")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_dataset_meta(path: Path) -> tuple[str | None, list[str] | None]:
    """Read instruction_field and metadata_fields from dataset.toml."""
    cfg = path / "dataset.toml"
    if not cfg.exists():
        return None, None
    raw = tomllib.loads(cfg.read_text())
    ds = raw.get("dataset", {})
    return ds.get("instruction_field"), ds.get("metadata_fields")


def _read_dataset_category(path: Path) -> str | None:
    """Read the ``category`` field from dataset.toml (e.g. ``"vlm"``)."""
    cfg = path / "dataset.toml"
    if not cfg.exists():
        return None
    raw = tomllib.loads(cfg.read_text())
    return raw.get("dataset", {}).get("category")


def _build_multimodal_instruction(text: str, row: dict, bench_dir: Path) -> list[dict]:
    """Construct OpenAI-format multimodal content blocks for VLM tasks.

    Picks up image paths from any row column (single string or list of
    strings) that points at a file under ``bench_dir/images/``. Each
    image becomes one ``{"type": "image_url", "image_url": {...}}`` block,
    encoded inline as a base64 data URI.
    """
    import base64
    import mimetypes

    blocks: list[dict] = [{"type": "text", "text": text}]
    seen: set[str] = set()
    for value in row.values():
        candidates: list[str] = []
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = [v for v in value if isinstance(v, str)]
        for cand in candidates:
            if not cand.startswith("images/"):
                continue
            if cand in seen:
                continue
            file_path = bench_dir / cand
            if not file_path.is_file():
                continue
            seen.add(cand)
            mime, _ = mimetypes.guess_type(file_path.name)
            if mime is None:
                mime = "image/png"
            data = base64.b64encode(file_path.read_bytes()).decode("ascii")
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    return blocks


def _render_instruction(path: Path, row: dict, instruction_field: str | None) -> str:
    """Render the instruction for a row.

    Priority: ``instruction.md.tpl`` template > ``instruction_field`` > full row text.
    """
    tpl_path = path / "instruction.md.tpl"
    if tpl_path.exists():
        tpl = tpl_path.read_text()
        try:
            return _render_simple_template(tpl, row)
        except Exception:
            pass
    if instruction_field and instruction_field in row:
        return str(row[instruction_field])
    return str(row.get("question", row.get("instruction", "")))


def _render_simple_template(tpl: str, row: dict) -> str:
    """Render ``{{field}}`` placeholders. Missing fields → empty string.

    List-valued fields (e.g. MCQ ``choices``) are formatted as a
    lettered block: ``(A) ...\n(B) ...\n...``.
    """
    import re

    def replace(match: re.Match[str]) -> str:
        field = match.group(1).strip()
        value = row.get(field, "")
        if isinstance(value, list):
            return _format_choices(value)
        return str(value)

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, tpl)


def _format_choices(items: list) -> str:
    """Format a list as a lettered MCQ block: ``(A) ...\n(B) ...``."""
    lines = []
    for i, item in enumerate(items):
        letter = chr(ord("A") + i)
        lines.append(f"({letter}) {item}")
    return "\n".join(lines)


def _build_metadata(row: dict, metadata_fields: list[str] | None) -> dict:
    """If metadata_fields is set, restrict to those keys; else pass full row."""
    if metadata_fields:
        return {k: row.get(k) for k in metadata_fields}
    return dict(row)


# ---------------------------------------------------------------------------
# Sandbox dataset (task-per-directory)
# ---------------------------------------------------------------------------


def _load_sandbox_dataset(
    path: Path,
    config: DatasetConfig,
    sandbox_backend: str | None,
    harness_name: str | None,
) -> BenchmarkResult:
    """Sandbox dataset: directory of task directories."""
    if config.tasks:
        task_dirs = [(path / ref.path).resolve() for ref in config.tasks]
    else:
        task_dirs = _discover_task_dirs(path)
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {path}")

    tasks = [_load_task_from_dir(td, benchmark_dir=path) for td in task_dirs]
    return BenchmarkResult(
        tasks=tasks,
        name=config.name,
        split=config.split,
        harness_name=harness_name or config.default_agent or "react",
        sandbox_backend=sandbox_backend or config.default_sandbox,
        description=config.description,
        category="agentic",
    )


def _load_single_task(
    path: Path,
    sandbox_backend: str | None,
    harness_name: str | None,
) -> BenchmarkResult:
    """Single task directory at the root."""
    task = _load_task_from_dir(path, benchmark_dir=path, sub_dir=None)
    return BenchmarkResult(
        tasks=[task],
        name=task.id,
        split="test",
        harness_name=harness_name or "react",
        sandbox_backend=sandbox_backend,
        description=task.metadata.get("task", {}).get("description", "") if isinstance(task.metadata.get("task"), dict) else "",
        category="agentic",
    )


def _load_auto_discover(
    path: Path,
    sandbox_backend: str | None,
    harness_name: str | None,
) -> BenchmarkResult:
    """Auto-discover subdirectories with task.toml."""
    task_dirs = _discover_task_dirs(path)
    if not task_dirs:
        raise FileNotFoundError(f"No task directories (with task.toml) found in {path}")
    tasks = [_load_task_from_dir(td, benchmark_dir=path) for td in task_dirs]
    return BenchmarkResult(
        tasks=tasks,
        name=path.name,
        split="test",
        harness_name=harness_name or "react",
        sandbox_backend=sandbox_backend,
        description=f"Auto-discovered tasks from {path.name}",
        category="agentic",
    )


def _load_task_from_dir(
    task_dir: Path,
    benchmark_dir: Path,
    sub_dir: Path | None | type = ...,
) -> Task:
    """Load a Harbor-style task directory into the new ``rllm.types.Task``.

    The whole task.toml goes into ``metadata``. Convenience keys (workdir,
    agent_user, verifier_user, verifier_timeout, etc.) are also lifted to
    top-level keys so :mod:`rllm.runner` can find them directly.
    """
    config_path = task_dir / "task.toml"
    if config_path.exists():
        raw = tomllib.loads(config_path.read_text())
    else:
        raw = {}

    instruction_path = task_dir / "instruction.md"
    instruction = instruction_path.read_text() if instruction_path.exists() else ""

    name = raw.get("task", {}).get("name", task_dir.name)

    # Lift commonly-used config into top-level metadata for the Runner
    metadata: dict = dict(raw)
    env_section = raw.get("environment", {}) or {}
    metadata["workdir"] = env_section.get("workdir", "/workspace")
    metadata["env_vars"] = env_section.get("env", {}) or {}
    metadata["agent_user"] = raw.get("agent", {}).get("user")
    metadata["verifier_user"] = raw.get("verifier", {}).get("user")
    metadata["verifier_timeout"] = raw.get("verifier", {}).get("timeout_sec", 600.0)
    metadata["agent_timeout"] = raw.get("agent", {}).get("timeout_sec", 600.0)
    rllm_section = raw.get("rllm", {}) or {}
    metadata["setup_commands"] = rllm_section.get("setup_commands", []) or []

    if sub_dir is ...:
        try:
            sub_dir = task_dir.relative_to(benchmark_dir) if task_dir != benchmark_dir else None
        except ValueError:
            sub_dir = None

    return Task(
        id=name,
        instruction=instruction,
        metadata=metadata,
        benchmark_dir=benchmark_dir,
        sub_dir=sub_dir,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_task_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "task.toml").exists())
