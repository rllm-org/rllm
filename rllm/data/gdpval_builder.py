"""Materialize OpenAI GDPval as rLLM sandbox tasks, in AA-compatible form.

The source dataset stores prompts plus Hugging Face paths for task inputs and
expert deliverables.  This builder stages the inputs at deterministic absolute
paths inside the solver environment, keeps expert deliverables under
``tests/reference`` on the host, writes the GDPval-AA v2 compatible sandbox
image, and renders Artificial Analysis' published solver prompt.

Grading is a single host-side stage: ``dataset.toml`` declares the multimodal
weighted-rubric ``reward_fn``, which runs after the solver exits. Tasks carry no
verifier of their own, so neither the rubric nor the expert deliverables ever
enter the sandbox.
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from rllm.data import gdpval_aa

logger = logging.getLogger(__name__)

REPO_ID = "openai/gdpval"

#: Grader written into the generated ``dataset.toml`` when the caller supplies
#: no catalog entry. Must stay equal to ``datasets.json``'s ``reward_fn`` for
#: ``gdpval`` — :func:`_write_dataset_toml` explains why.
DEFAULT_REWARD_FN = "gdpval_rubric_reward_fn"

OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_EXTERNAL_OPEN_REL = re.compile(
    rb"<Relationship\b(?P<attrs>[^<>]*\bTargetMode=(?:\"External\"|'External')[^<>]*?)(?<!/)>",
    re.IGNORECASE,
)
_SUSPICIOUS_SETTINGS = re.compile(rb"(?:xmlns:ns\d+\s*=|</?ns\d+:)")

#: Where the harness stages and reports the solver's submission; keep them in
#: sync with :mod:`rllm.harnesses.stirrup`, which writes and then collects them.
SUBMISSION_ROOT = "/tmp/gdpval-aa"
SUBMISSION_DIR = f"{SUBMISSION_ROOT}/submission"
RUN_METADATA_PATH = f"{SUBMISSION_ROOT}/run.json"


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relationship_source(name: str) -> str | None:
    if name == "_rels/.rels":
        return ""
    marker = "/_rels/"
    if marker not in name or not name.endswith(".rels"):
        return None
    parent, leaf = name.split(marker, 1)
    return posixpath.join(parent, leaf[: -len(".rels")])


def _validate_office_package(path: Path) -> dict:
    report: dict = {
        "zip_ok": False,
        "required_parts_missing": [],
        "xml_errors": [],
        "missing_relationship_targets": [],
        "valid": False,
    }
    try:
        with zipfile.ZipFile(path) as package:
            names = set(package.namelist())
            bad_member = package.testzip()
            if bad_member:
                report["xml_errors"].append(f"CRC failure: {bad_member}")
                return report
            report["zip_ok"] = True
            report["required_parts_missing"] = sorted({"[Content_Types].xml", "_rels/.rels"} - names)
            for name in sorted(names):
                if not (name.endswith(".xml") or name.endswith(".rels")):
                    continue
                try:
                    root = ElementTree.fromstring(package.read(name))
                except (ElementTree.ParseError, UnicodeError) as exc:
                    report["xml_errors"].append(f"{name}: {exc}")
                    continue
                if not name.endswith(".rels"):
                    continue
                source = _relationship_source(name)
                if source is None:
                    continue
                source_dir = posixpath.dirname(source)
                for relationship in root.findall(f"{{{_REL_NS}}}Relationship"):
                    if relationship.get("TargetMode", "").lower() == "external":
                        continue
                    target = relationship.get("Target")
                    if not target or target.startswith("/"):
                        continue
                    resolved = posixpath.normpath(posixpath.join(source_dir, target))
                    if resolved not in names:
                        report["missing_relationship_targets"].append(f"{name} -> {resolved}")
    except (OSError, zipfile.BadZipFile) as exc:
        report["xml_errors"].append(str(exc))
        return report

    report["valid"] = not (report["required_parts_missing"] or report["xml_errors"] or report["missing_relationship_targets"])
    return report


def _repair_docx_members(package: zipfile.ZipFile) -> tuple[list[tuple[zipfile.ZipInfo, bytes]], list[str]]:
    members = [(info, package.read(info.filename)) for info in package.infolist()]
    by_name = {info.filename: data for info, data in members}
    changes: list[str] = []
    remove_settings = bool(_SUSPICIOUS_SETTINGS.search(by_name.get("word/settings.xml", b"")))
    if remove_settings:
        changes.append("removed suspicious word/settings.xml")

    repaired: list[tuple[zipfile.ZipInfo, bytes]] = []
    for info, data in members:
        if remove_settings and info.filename == "word/settings.xml":
            continue
        if info.filename.endswith(".rels"):
            data, count = _EXTERNAL_OPEN_REL.subn(rb"<Relationship\g<attrs>/>", data)
            if count:
                changes.append(f"closed {count} malformed external relationship(s) in {info.filename}")
            if remove_settings and info.filename == "word/_rels/document.xml.rels":
                try:
                    root = ElementTree.fromstring(data)
                    removed = 0
                    for relationship in list(root):
                        if relationship.get("Type", "").endswith("/settings") or relationship.get("Target") == "settings.xml":
                            root.remove(relationship)
                            removed += 1
                    if removed:
                        ElementTree.register_namespace("", _REL_NS)
                        data = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
                        changes.append(f"removed {removed} settings relationship(s) from {info.filename}")
                except ElementTree.ParseError:
                    pass
        repaired.append((info, data))
    return repaired, changes


def _repair_office_file(path: Path, backup_path: Path) -> dict:
    original_hash = _sha256(path)
    before = _validate_office_package(path) if path.suffix.lower() in OFFICE_EXTENSIONS else None
    result = {
        "file": path.name,
        "status": "skipped",
        "original_sha256": original_hash,
        "repaired_sha256": original_hash,
        "backup": None,
        "changes": [],
        "validation_before": before,
        "validation_after": before,
    }
    if path.suffix.lower() not in OFFICE_EXTENSIONS:
        return result
    if path.suffix.lower() != ".docx":
        result["status"] = "valid" if before and before["valid"] else "unresolved"
        return result
    try:
        with zipfile.ZipFile(path) as package:
            members, changes = _repair_docx_members(package)
    except (OSError, zipfile.BadZipFile):
        result["status"] = "unresolved"
        return result
    if not changes:
        result["status"] = "valid" if before and before["valid"] else "unresolved"
        return result

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w") as target:
            for info, data in members:
                target.writestr(info, data)
        after = _validate_office_package(temporary)
        if not after["valid"]:
            result.update(status="unresolved", changes=changes, backup=str(backup_path), validation_after=after)
            return result
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    result.update(
        status="repaired",
        repaired_sha256=_sha256(path),
        backup=str(backup_path),
        changes=changes,
        validation_after=after,
    )
    return result


def _download_files(paths: list[str], destination: Path) -> list[str]:
    """Copy dataset files out of the HF cache into *destination*.

    Repairs mutate the staged copy only — the cache stays pristine so a rebuild
    always starts from the dataset as published.
    """
    from huggingface_hub import hf_hub_download

    destination.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for repo_path in paths:
        cached = Path(hf_hub_download(REPO_ID, str(repo_path), repo_type="dataset"))
        target = destination / Path(str(repo_path)).name
        shutil.copy2(cached, target)
        names.append(target.name)
    return names


def _repair_staged_files(paths: list[Path], *, task_id: str, role: str, backup_root: Path) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        if path.suffix.lower() not in OFFICE_EXTENSIONS:
            continue
        record = _repair_office_file(path, backup_root / task_id / role / path.name)
        record.update(task_id=task_id, role=role, staged_path=str(path))
        records.append(record)
    return records


def _dataset_revision() -> str:
    try:
        from huggingface_hub import dataset_info

        return str(dataset_info(REPO_ID).sha or "")
    except Exception:
        logger.debug("[gdpval] could not resolve dataset revision", exc_info=True)
        return ""


def reference_file_paths(input_names: list[str]) -> list[str]:
    """Absolute in-sandbox paths for staged reference files.

    ``environment/files/`` is uploaded to the task's workdir, so a staged file
    lands at ``<workdir>/<basename>``. The prompt must quote these exact paths.
    """
    return [f"{gdpval_aa.AA_WORKDIR}/{name}" for name in input_names]


def _write_instruction(task_dir: Path, row: dict, input_names: list[str]) -> str:
    """Write AA's published task prompt. Returns the rendered prompt."""
    prompt = gdpval_aa.render_aa_gdpval_task_prompt(
        task_description=str(row.get("prompt") or "").strip(),
        reference_paths=reference_file_paths(input_names),
    )
    (task_dir / "instruction.md").write_text(prompt, encoding="utf-8")
    return prompt


def _image_lines() -> list[str]:
    """The ``[environment]`` image declaration, in pull or build mode.

    **Pull** (a published image is pinned): the image *is* the environment, so
    ``replay_dockerfile = false`` — every backend boots it as-is. This is how the
    rest of rLLM's benchmarks work, and it is the mode that makes GDPval usable
    at scale: no per-environment build, and every run provably uses the same
    closure by digest.

    **Build** (nothing published): the declared image is only the Dockerfile's
    base, so replay must stay on. Declaring an image otherwise defaults replay
    *off* — the Harbor convention, where a declared image is already built —
    which leaves non-docker backends booting bare Debian with no solver user, no
    packages and no venv, while the snapshot still reports success.
    """
    published = gdpval_aa.published_image_ref()
    if published:
        return [
            f'docker_image = "{published}"',
            "# The published image is the whole environment; nothing to replay.",
            "replay_dockerfile = false",
        ]
    return [
        f'docker_image = "{gdpval_aa.AA_BASE_IMAGE}@{gdpval_aa.AA_BASE_IMAGE_DIGEST}"',
        "# Base image only — the AA closure is the Dockerfile's RUN steps.",
        "replay_dockerfile = true",
    ]


def _write_task_toml(task_dir: Path, row: dict, input_names: list[str]) -> None:
    """Write the harbor task config.

    No ``[verifier]`` block: grading is the dataset-level ``reward_fn``
    :func:`_write_dataset_toml` declares, which runs on the host after the
    sandbox is gone. A per-task verifier here would be a second grader that
    disagrees with the first, and whichever one the CLI happened to resolve
    would decide the score.

    Expert deliverables are deliberately absent: they stay under ``tests/`` on
    the host and are never uploaded, since nothing in the sandbox reads them.
    """
    lines = [
        'schema_version = "1.1"',
        f'task_id = "{row["task_id"]}"',
        f'sector = """{_toml_escape(str(row.get("sector") or ""))}"""',
        f'occupation = """{_toml_escape(str(row.get("occupation") or ""))}"""',
        f"reference_files = {json.dumps(reference_file_paths(input_names))}",
        # The loader lifts these into ``task.metadata``, where the opt-in
        # The ``gdpval`` reward_fn reads them: the raw work request (the AA
        # instruction wraps it in submission boilerplate the judge should not
        # weigh) and GDPval's weighted pass/fail criteria.
        #
        # ``task.toml`` is never uploaded to the sandbox — only
        # ``environment/files/`` is — so the rubric, which states the expected
        # answers, stays out of the solver's reach. It is deliberately NOT
        # written to ``tests/rubric.json`` the way the cookbook does for its
        # in-sandbox grader: ``tests/`` *is* uploaded, and a host-side grader
        # has no use for a copy in there.
        f'prompt = """{_toml_escape(str(row.get("prompt") or ""))}"""',
        f'rubric_json = """{_toml_escape(str(row.get("rubric_json") or "[]"))}"""',
        "",
        "[task]",
        f'name = "{row["task_id"]}"',
        "",
        "[environment]",
        f'workdir = "{gdpval_aa.AA_WORKDIR}"',
        *_image_lines(),
        "cpus = 4",
        "memory_mb = 16384",
        "storage_mb = 32768",
        "",
        "[agent]",
        # AA runs the solver as the non-root user the task prompt names.
        f'user = "{gdpval_aa.AA_AGENT_USER}"',
        "timeout_sec = 14400",
        "",
    ]
    (task_dir / "task.toml").write_text("\n".join(lines), encoding="utf-8")


def _write_provenance(
    task_dir: Path,
    row: dict,
    *,
    prompt: str,
    input_names: list[str],
    inputs_dir: Path,
    dataset_revision: str,
) -> None:
    """Record everything a later grading stage needs to trust this run."""
    reference_files = []
    for name, path in zip(input_names, reference_file_paths(input_names), strict=True):
        staged = inputs_dir / name
        reference_files.append({"path": path, "sha256": _sha256(staged), "size_bytes": staged.stat().st_size})

    provenance = {
        "schema_version": 1,
        "benchmark": "gdpval",
        "methodology": "GDPval-AA v2",
        "task_id": row["task_id"],
        "sector": row.get("sector", ""),
        "occupation": row.get("occupation", ""),
        "dataset_repo": REPO_ID,
        "dataset_revision": dataset_revision,
        "sandbox_base_image": f"{gdpval_aa.AA_BASE_IMAGE}@{gdpval_aa.AA_BASE_IMAGE_DIGEST}",
        "sandbox_image_digest": gdpval_aa.AA_BASE_IMAGE_DIGEST,
        "sandbox_platform": gdpval_aa.AA_PLATFORM,
        "debian_snapshot": gdpval_aa.AA_DEBIAN_SNAPSHOT,
        "system_prompt_sha256": gdpval_aa.sha256_text(gdpval_aa.AA_GDPVAL_SYSTEM_PROMPT),
        "task_prompt_sha256": gdpval_aa.sha256_text(prompt),
        "reference_files": reference_files,
    }
    (task_dir / "gdpval_aa.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_dataset_toml(out: Path, *, name: str, split: str, description: str, default_agent: str, reward_fn: str) -> None:
    """Write ``dataset.toml``, including the shared ``[verifier]``.

    The ``[verifier] name`` must match the catalog's ``reward_fn``. The eval CLI
    resolves a benchmark two ways — from the catalog by name, or from this file
    once the directory exists on disk — and both must land on the same grader.
    Leaving it out is what makes a run silently fall through to whatever
    per-task verifier it can find.
    """
    (out / "dataset.toml").write_text(
        "\n".join(
            [
                "[dataset]",
                f'name = "{name}"',
                'type = "sandbox"',
                f'description = "{_toml_escape(description)}"',
                'default_sandbox = "docker"',
                f'default_agent = "{default_agent}"',
                f'split = "{split}"',
                "",
                "[verifier]",
                f'name = "{reward_fn}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_benchmark(
    *,
    name: str = "gdpval",
    split: str = "train",
    out_dir: str | Path,
    catalog_entry: dict | None = None,
    limit: int | None = None,
    occupations: list[str] | None = None,
    default_agent: str = "stirrup",
    repair_office_files: bool = True,
    clean: bool = False,
    register: bool = True,
) -> Path:
    """Download and materialize the official GDPval split."""
    from datasets import load_dataset

    if catalog_entry:
        split = catalog_entry.get("eval_split") or split
        default_agent = catalog_entry.get("default_agent") or default_agent

    out = Path(out_dir).expanduser()
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [dict(row) for row in load_dataset(REPO_ID, split=split)]
    if occupations:
        allowed = {value.casefold().strip() for value in occupations}
        rows = [row for row in rows if str(row.get("occupation", "")).casefold().strip() in allowed]
    if limit is not None:
        rows = rows[:limit]

    # Pull mode ships a three-line wrapper over the published image; build
    # mode ships the full recipe so the closure can be constructed locally.
    dockerfile_body = gdpval_aa.render_task_dockerfile() if gdpval_aa.published_image_ref() else gdpval_aa.render_dockerfile()
    dataset_revision = _dataset_revision()
    repair_records: list[dict] = []
    registry_rows: list[dict] = []
    backup_root = out / ".office-repair-originals"
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        task_dir = out / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
        inputs_dir = task_dir / "environment" / "files"
        gold_dir = task_dir / "tests" / "reference"
        input_names = _download_files(list(row.get("reference_files") or []), inputs_dir)
        gold_names = _download_files(list(row.get("deliverable_files") or []), gold_dir)
        if repair_office_files:
            repair_records.extend(_repair_staged_files([inputs_dir / name for name in input_names], task_id=task_id, role="input", backup_root=backup_root))
            repair_records.extend(_repair_staged_files([gold_dir / name for name in gold_names], task_id=task_id, role="expert", backup_root=backup_root))

        (task_dir / "environment" / "Dockerfile").write_text(dockerfile_body, encoding="utf-8")
        prompt = _write_instruction(task_dir, row, input_names)
        _write_task_toml(task_dir, row, input_names)
        _write_provenance(task_dir, row, prompt=prompt, input_names=input_names, inputs_dir=inputs_dir, dataset_revision=dataset_revision)
        registry_rows.append(
            {
                "task_id": task_id,
                "id": task_id,
                "instruction": prompt,
                "task_path": str(task_dir),
                "occupation": row.get("occupation", ""),
                "sector": row.get("sector", ""),
            }
        )

    if not registry_rows:
        raise RuntimeError(f"no GDPval tasks materialized from {REPO_ID} split={split}")

    description = (catalog_entry or {}).get("description") or "GDPval: 220 economically valuable knowledge-work tasks with expert deliverables."
    reward_fn = (catalog_entry or {}).get("reward_fn") or DEFAULT_REWARD_FN
    _write_dataset_toml(out, name=name, split=split, description=description, default_agent=default_agent, reward_fn=reward_fn)
    if repair_office_files:
        statuses: dict[str, int] = {}
        for record in repair_records:
            statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        manifest = {
            "schema_version": 1,
            "policy": "Known GDPval DOCX repairs only; unknown defects are reported as unresolved.",
            "summary": {"files": len(repair_records), "statuses": statuses},
            "files": repair_records,
        }
        (out / "office_repair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if register:
        try:
            from rllm.data import DatasetRegistry

            DatasetRegistry.register_dataset(
                name=name,
                data=registry_rows,
                split=split,
                source=REPO_ID,
                description=description,
                category=(catalog_entry or {}).get("category", "agentic"),
            )
        except Exception:
            logger.warning("[gdpval] could not register rows in DatasetRegistry", exc_info=True)
    return out
