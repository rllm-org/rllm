"""Tests for the cloudflared tunnel helper.

We don't actually spawn cloudflared — we monkey-patch ``subprocess.Popen``
so the tests are hermetic and run in milliseconds.
"""

from __future__ import annotations

import io
import subprocess

import pytest

from rllm.eval import tunnel
from rllm.eval.tunnel import TunnelError, start_cloudflared_tunnel, stop_tunnel


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` that emits a scripted stdout stream."""

    def __init__(self, output_lines: list[str], *, exits_with: int | None = None) -> None:
        self.stdout = io.StringIO("".join(output_lines))
        self._exits_with = exits_with
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._exits_with if self._exits_with is not None else self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _patch_popen(monkeypatch, ctor) -> list[tuple]:
    """Capture every Popen call and route it to *ctor*."""
    calls: list[tuple] = []

    def _fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return ctor(args, kwargs)

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return calls


def test_extracts_url_from_stdout(monkeypatch):
    output = [
        "2024-01-01 INFO Starting tunnel\n",
        "Your quick tunnel is now visible at:\n",
        "  https://abc-def-ghi.trycloudflare.com\n",
        "Connection registered\n",
    ]
    _patch_popen(monkeypatch, lambda args, kw: _FakePopen(output))

    url, proc = start_cloudflared_tunnel(8000)

    assert url == "https://abc-def-ghi.trycloudflare.com"
    assert isinstance(proc, _FakePopen)


def test_args_include_target_port(monkeypatch):
    calls = _patch_popen(monkeypatch, lambda args, kw: _FakePopen(["https://x.trycloudflare.com\n"]))

    start_cloudflared_tunnel(54321)

    args, _ = calls[0]
    assert args == ["cloudflared", "tunnel", "--url", "http://localhost:54321"]


def test_missing_cloudflared_binary_raises_with_install_hint(monkeypatch):
    def _raise(*_a, **_kw):
        raise FileNotFoundError("cloudflared")

    monkeypatch.setattr(subprocess, "Popen", _raise)

    with pytest.raises(TunnelError, match="brew install cloudflared|--gateway-public-url"):
        start_cloudflared_tunnel(8000, timeout=1.0)


def test_premature_exit_raises(monkeypatch):
    """If cloudflared dies before printing a URL, surface the exit code."""

    def ctor(args, kw):
        return _FakePopen([], exits_with=2)

    _patch_popen(monkeypatch, ctor)

    with pytest.raises(TunnelError, match="exited with code 2"):
        start_cloudflared_tunnel(8000, timeout=1.0)


def test_timeout_terminates_proc(monkeypatch):
    """When no URL appears within timeout we must kill the proc, not leak it."""
    holder: dict[str, _FakePopen] = {}

    def ctor(args, kw):
        # Empty pipe with no exit — readline blocks forever in real life, but
        # io.StringIO("") returns "" immediately, which the helper interprets
        # as "pipe closed → process exited" and falls through to poll().
        # To simulate "still running, no output", we wrap a stream that
        # yields a few empty lines then EOF, and override poll() to None.
        proc = _FakePopen([])
        # Force "still running" so the timeout branch is exercised.
        proc._exits_with = None  # poll() returns None
        # poll() returns None → empty readline → loop continues → timeout hit.
        holder["proc"] = proc
        return proc

    _patch_popen(monkeypatch, ctor)
    monkeypatch.setattr(tunnel, "_TUNNEL_URL_RE", tunnel._TUNNEL_URL_RE)  # noop

    # Tight timeout so the test finishes fast.
    with pytest.raises(TunnelError, match="did not surface a public URL"):
        start_cloudflared_tunnel(8000, timeout=0.1)
    assert holder["proc"].terminated is True


def test_stop_tunnel_terminates_running_proc():
    proc = _FakePopen([])
    proc._exits_with = None  # still running

    stop_tunnel(proc)

    assert proc.terminated is True


def test_stop_tunnel_is_noop_when_already_exited():
    proc = _FakePopen([])
    proc._exits_with = 0  # already exited

    stop_tunnel(proc)

    assert proc.terminated is False  # we didn't try to terminate a dead proc


def test_stop_tunnel_handles_none():
    """Manager calls stop_tunnel(self._tunnel_proc) where the field may be None."""
    stop_tunnel(None)  # must not raise
