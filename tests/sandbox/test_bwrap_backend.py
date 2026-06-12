"""Tests for BwrapSandbox backend.

These tests require bwrap (bubblewrap) to be installed. They are skipped on
platforms where bwrap is not available (macOS, CI without bubblewrap).
"""

import os
import subprocess
import tempfile

import pytest

from rllm.sandbox.backends.bwrap import BwrapSandbox, is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="bwrap not installed")


class TestBwrapSandbox:
    def test_basic_exec(self):
        sandbox = BwrapSandbox(name="test-basic")
        try:
            result = sandbox.exec("echo hello")
            assert result.strip() == "hello"
        finally:
            sandbox.close()

    def test_network_isolated(self):
        sandbox = BwrapSandbox(name="test-net")
        try:
            with pytest.raises(subprocess.CalledProcessError):
                sandbox.exec("ping -c 1 -W 1 8.8.8.8", timeout=5)
        finally:
            sandbox.close()

    def test_host_secrets_not_visible(self):
        """Verify /proc/1/environ is not accessible (AWS creds protection)."""
        sandbox = BwrapSandbox(name="test-secrets")
        try:
            with pytest.raises(subprocess.CalledProcessError):
                sandbox.exec("cat /proc/1/environ")
        finally:
            sandbox.close()

    def test_app_dir_writable(self):
        sandbox = BwrapSandbox(name="test-app")
        try:
            sandbox.exec("echo 'test content' > /app/test.txt")
            result = sandbox.exec("cat /app/test.txt")
            assert result.strip() == "test content"
        finally:
            sandbox.close()

    def test_tmp_dir_writable(self):
        sandbox = BwrapSandbox(name="test-tmp")
        try:
            sandbox.exec("echo 'tmp data' > /tmp/test.txt")
            result = sandbox.exec("cat /tmp/test.txt")
            assert result.strip() == "tmp data"
        finally:
            sandbox.close()

    def test_upload_file_to_app(self):
        sandbox = BwrapSandbox(name="test-upload")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("uploaded content")
                f.flush()
                sandbox.upload_file(f.name, "/app/uploaded.txt")
            result = sandbox.exec("cat /app/uploaded.txt")
            assert result.strip() == "uploaded content"
        finally:
            sandbox.close()
            os.unlink(f.name)

    def test_upload_file_invalid_path_raises(self):
        sandbox = BwrapSandbox(name="test-invalid-path")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("data")
                f.flush()
            with pytest.raises(ValueError, match="must be under /tmp or /app"):
                sandbox.upload_file(f.name, "/home/user/evil.txt")
        finally:
            sandbox.close()
            os.unlink(f.name)

    def test_timeout(self):
        sandbox = BwrapSandbox(name="test-timeout")
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                sandbox.exec("sleep 60", timeout=1)
        finally:
            sandbox.close()

    def test_nonzero_exit_raises(self):
        sandbox = BwrapSandbox(name="test-exit")
        try:
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                sandbox.exec("exit 42")
            assert exc_info.value.returncode == 42
        finally:
            sandbox.close()

    def test_close_idempotent(self):
        sandbox = BwrapSandbox(name="test-close")
        sandbox.close()
        sandbox.close()  # Should not raise

    def test_python_execution(self):
        sandbox = BwrapSandbox(name="test-python")
        try:
            result = sandbox.exec("python3 -c 'print(2 + 2)'")
            assert result.strip() == "4"
        finally:
            sandbox.close()
