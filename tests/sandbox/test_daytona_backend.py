"""Tests for the Daytona sandbox backend wrapper.

Covers behavior that doesn't require the real ``daytona`` SDK (or a
DAYTONA_API_KEY): currently the friendly error when the SDK isn't
installed.
"""

from __future__ import annotations

import builtins
import sys

import pytest

from rllm.sandbox.backends.daytona import DaytonaSandbox


def test_missing_daytona_sdk_raises_friendly_install_hint(monkeypatch):
    """When the ``daytona`` package isn't installed, instantiating
    DaytonaSandbox should raise an ImportError naming the install command,
    not a bare ``ModuleNotFoundError("No module named 'daytona'")``.
    """
    # Drop any cached daytona module and intercept the lazy import.
    monkeypatch.delitem(sys.modules, "daytona", raising=False)
    real_import = builtins.__import__

    def _block_daytona(name, *args, **kwargs):
        if name == "daytona" or name.startswith("daytona."):
            raise ModuleNotFoundError("No module named 'daytona'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_daytona)

    with pytest.raises(ImportError, match="pip install daytona"):
        DaytonaSandbox(name="test")
