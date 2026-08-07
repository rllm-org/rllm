"""Pinned MCP-Atlas source bootstrap shared by the harness and scorer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

from rllm import paths
from rllm.integrations.mcp_atlas.constants import SOURCE_REVISION, SOURCE_URL

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()
_SCORER_MODULE: ModuleType | None = None


def cache_dir() -> Path:
    return Path(paths.rllm_path("integrations", "mcp-atlas", SOURCE_REVISION))


def patch_path() -> Path:
    return Path(__file__).with_name("session_base_url.patch")


def patch_sha256() -> str:
    return hashlib.sha256(patch_path().read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    logger.debug("MCP-Atlas bootstrap command: %s", " ".join(command[:3]))
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)


def ensure_source() -> Path:
    """Return a verified checkout with the rLLM session-routing patch applied."""
    target = cache_dir()
    marker = target / ".rllm-source.json"
    expected = {"source_revision": SOURCE_REVISION, "patch_sha256": patch_sha256()}
    with _LOCK:
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")) == expected:
                    head = _run(["git", "rev-parse", "HEAD"], cwd=target).stdout.strip()
                    if head == SOURCE_REVISION:
                        _run(["git", "apply", "--reverse", "--check", str(patch_path())], cwd=target)
                        return target
            except (OSError, ValueError, subprocess.SubprocessError):
                logger.warning("MCP-Atlas cache verification failed; rebuilding %s", target)

        if target.exists():
            # The target is a narrow, versioned cache directory owned by this
            # integration. Avoid recursive deletion: git clean/reset restores it.
            if not (target / ".git").is_dir():
                raise RuntimeError(f"MCP-Atlas cache path exists but is not a git checkout: {target}")
            _run(["git", "reset", "--hard"], cwd=target)
            _run(["git", "clean", "-fd"], cwd=target)
            _run(["git", "fetch", "origin", SOURCE_REVISION], cwd=target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--no-checkout", SOURCE_URL, str(target)])

        _run(["git", "checkout", "--detach", SOURCE_REVISION], cwd=target)
        _run(["git", "apply", "--check", str(patch_path())], cwd=target)
        _run(["git", "apply", str(patch_path())], cwd=target)
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        return target


def ensure_harness_build() -> Path:
    source = ensure_source()
    harness_dir = source / "services" / "agent-harness"
    marker = harness_dir / ".rllm-build.json"
    lock_digest = hashlib.sha256((harness_dir / "package-lock.json").read_bytes()).hexdigest()
    expected = {"package_lock_sha256": lock_digest, "patch_sha256": patch_sha256()}
    with _LOCK:
        if marker.is_file() and (harness_dir / "dist" / "index.js").is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")) == expected:
                    return harness_dir
            except (OSError, ValueError):
                pass
        _run(["npm", "ci"], cwd=harness_dir)
        _run(["npm", "run", "build"], cwd=harness_dir)
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return harness_dir


def load_official_scorer() -> ModuleType:
    """Import the scorer from the verified checkout exactly once."""
    global _SCORER_MODULE
    with _LOCK:
        if _SCORER_MODULE is not None:
            return _SCORER_MODULE
        scorer_path = ensure_source() / "services" / "scoring" / "score_claims.py"
        # Prevent matplotlib from attempting to create a cache in an unwritable home.
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir() / ".matplotlib"))
        spec = importlib.util.spec_from_file_location("rllm_pinned_mcp_atlas_score_claims", scorer_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import pinned MCP-Atlas scorer from {scorer_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            raise RuntimeError("MCP-Atlas scorer dependencies are missing; install rllm[mcp-atlas]") from exc
        _SCORER_MODULE = module
        return module


__all__ = ["cache_dir", "ensure_harness_build", "ensure_source", "load_official_scorer", "patch_sha256"]
