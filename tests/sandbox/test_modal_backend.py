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


class _CreateWithTags:
    @staticmethod
    def create(*args, tags=None, **kwargs): ...


class _CreateWithoutTags:
    @staticmethod
    def create(*args, app=None, image=None, timeout=None, name=None, **kwargs): ...


def test_supports_create_tags_feature_detection():
    """Older modal SDKs (< 1.5, no ``tags`` kwarg) must not be sent one —
    passing it makes every sandbox create fail with a TypeError."""
    from rllm.sandbox.backends.modal_backend import _supports_create_tags

    assert _supports_create_tags(_CreateWithTags) is True
    assert _supports_create_tags(_CreateWithoutTags) is False


def test_attach_run_tags_swallows_set_tags_failure():
    from rllm.sandbox.backends.modal_backend import _attach_run_tags

    class Boom:
        def set_tags(self, tags):
            raise RuntimeError("no tags api")

    _attach_run_tags(Boom(), {"rllm_run_id": "x"}, "sb")  # must not raise

    seen = {}

    class Ok:
        def set_tags(self, tags):
            seen.update(tags)

    _attach_run_tags(Ok(), {"rllm_run_id": "x"}, "sb")
    assert seen == {"rllm_run_id": "x"}
