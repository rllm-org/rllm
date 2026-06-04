import importlib
import logging

import pytest

import rllm
from rllm.utils.logging import configure_logging_from_env, get_log_level_from_env


@pytest.fixture(autouse=True)
def restore_logging_state():
    root = logging.getLogger()
    logger = logging.getLogger("rllm")
    root_handlers = list(root.handlers)
    root_level = root.level
    logger_level = logger.level

    yield

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in root_handlers:
        root.addHandler(handler)
    root.setLevel(root_level)
    logger.setLevel(logger_level)


def test_get_log_level_from_env_uses_default(monkeypatch):
    monkeypatch.delenv("RLLM_LOG_LEVEL", raising=False)

    assert get_log_level_from_env() == logging.WARNING
    assert get_log_level_from_env(default="WARNING") == logging.WARNING


def test_get_log_level_from_env_reads_level(monkeypatch):
    monkeypatch.setenv("RLLM_LOG_LEVEL", "debug")

    assert get_log_level_from_env() == logging.DEBUG


def test_get_log_level_from_env_falls_back_for_invalid_level(monkeypatch):
    monkeypatch.setenv("RLLM_LOG_LEVEL", "not-a-level")

    assert get_log_level_from_env(default="ERROR") == logging.ERROR


def test_configure_logging_from_env_sets_rllm_logger(monkeypatch):
    logger = logging.getLogger("rllm")
    monkeypatch.setenv("RLLM_LOG_LEVEL", "ERROR")

    assert configure_logging_from_env() == logging.ERROR
    assert logger.level == logging.ERROR


def test_configure_logging_from_env_initializes_basic_logging(monkeypatch):
    root = logging.getLogger()
    logger = logging.getLogger("rllm")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    monkeypatch.setenv("RLLM_LOG_LEVEL", "WARNING")

    assert configure_logging_from_env() == logging.WARNING
    assert root.handlers
    assert root.level == logging.WARNING
    assert logger.level == logging.WARNING


def test_import_configures_rllm_logger_from_env(monkeypatch):
    logger = logging.getLogger("rllm")
    monkeypatch.setenv("RLLM_LOG_LEVEL", "CRITICAL")

    importlib.reload(rllm)

    assert logger.level == logging.CRITICAL
