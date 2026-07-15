"""Tests for the Sailboxes sandbox backend wrapper.

Covers behavior that doesn't require the real ``sail`` SDK (or a
SAIL_API_KEY): the friendly error when the SDK isn't installed, the pure
command-building / image-classification helpers, and the deliberately-inert
prune/delete probes (Sailboxes exposes no checkpoint GET/delete API).
"""

from __future__ import annotations

import builtins
import sys

import pytest

from rllm.sandbox.backends.sailboxes import (
    _build_exec_command,
    _looks_like_docker_image,
    _sailboxes_ref_absent,
    delete_sailboxes_snapshot,
)


def test_missing_sail_sdk_raises_friendly_install_hint(monkeypatch):
    """When the ``sail`` package isn't installed, instantiating
    SailboxesSandbox should raise an ImportError naming the install command,
    not a bare ``ModuleNotFoundError("No module named 'sail'")``.
    """
    from rllm.sandbox.backends.sailboxes import SailboxesSandbox

    # Drop any cached sail module and intercept the lazy import.
    monkeypatch.delitem(sys.modules, "sail", raising=False)
    real_import = builtins.__import__

    def _block_sail(name, *args, **kwargs):
        if name == "sail" or name.startswith("sail."):
            raise ModuleNotFoundError("No module named 'sail'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_sail)

    with pytest.raises(ImportError, match="pip install sail"):
        SailboxesSandbox(name="test")


# --- _build_exec_command: pins su user-switch + persistent-env behavior, the
# same contract as the Modal/Daytona backends, without a live Sailbox. --------


def test_plain_command_unchanged():
    assert _build_exec_command("echo hi", None, None) == "echo hi"
    assert _build_exec_command("echo hi", {}, None) == "echo hi"


def test_persistent_env_is_exported_first():
    assert _build_exec_command("echo hi", {"FOO": "bar"}, None) == "export FOO=bar; echo hi"


def test_persistent_env_values_are_quoted():
    assert _build_exec_command("run", {"K": "a b"}, None) == "export K='a b'; run"


def test_user_switch_with_name():
    assert _build_exec_command("echo hi", None, "tester") == "su tester -s /bin/bash -c 'echo hi'"


def test_user_switch_with_uid_resolves_name():
    assert _build_exec_command("echo hi", None, 1000) == "su $(getent passwd 1000 | cut -d: -f1) -s /bin/bash -c 'echo hi'"


def test_env_is_applied_inside_the_switched_shell():
    # The exports must live inside the su'd shell so they reach the target user.
    assert _build_exec_command("echo hi", {"FOO": "bar"}, "tester") == "su tester -s /bin/bash -c 'export FOO=bar; echo hi'"


# --- image classification + inert prune/delete -------------------------------


@pytest.mark.parametrize(
    "image,is_docker",
    [
        ("python:3.11-slim", True),
        ("ghcr.io/foo/bar:tag", True),
        ("registry.io/team/img", True),
        ("rllm-env-abc123def456", False),  # a checkpoint id / bare token
        ("ckpt_0a1b2c3d", False),
    ],
)
def test_looks_like_docker_image(image, is_docker):
    """``:`` or ``/`` ⇒ Docker ref (Debian fallback); bare token ⇒ checkpoint."""
    assert _looks_like_docker_image(image) is is_docker


def test_prune_and_delete_are_conservative_noops():
    """Without a checkpoint GET/delete API, absence can't be confirmed and
    deletion can't be performed: never prune, never claim deletion (checkpoints
    self-expire via their TTL).
    """
    assert _sailboxes_ref_absent("any-ref") is False
    assert delete_sailboxes_snapshot("any-ref") is False
