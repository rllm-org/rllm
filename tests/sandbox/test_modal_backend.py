"""Unit tests for the Modal backend's exec command construction.

``_build_exec_command`` is the pure string transform behind
:meth:`ModalSandbox.exec`; testing it here pins the user-switch (``su``) and
persistent-env behavior that aligns rLLM's Modal path with Harbor — without
needing a live Modal sandbox.
"""

from __future__ import annotations

from rllm.sandbox.backends.modal_backend import _build_exec_command


def test_plain_command_unchanged():
    assert _build_exec_command("echo hi", None, None) == "echo hi"
    assert _build_exec_command("echo hi", {}, None) == "echo hi"


def test_persistent_env_is_exported_first():
    out = _build_exec_command("echo hi", {"FOO": "bar"}, None)
    assert out == "export FOO=bar; echo hi"


def test_persistent_env_values_are_quoted():
    out = _build_exec_command("run", {"K": "a b"}, None)
    assert out == "export K='a b'; run"


def test_user_switch_with_name():
    out = _build_exec_command("echo hi", None, "tester")
    assert out == "su tester -s /bin/bash -c 'echo hi'"


def test_user_switch_with_uid_resolves_name():
    out = _build_exec_command("echo hi", None, 1000)
    assert out == "su $(getent passwd 1000 | cut -d: -f1) -s /bin/bash -c 'echo hi'"


def test_env_is_applied_inside_the_switched_shell():
    out = _build_exec_command("echo hi", {"FOO": "bar"}, "tester")
    # The exports must live inside the su'd shell so they reach the target user.
    assert out == "su tester -s /bin/bash -c 'export FOO=bar; echo hi'"
