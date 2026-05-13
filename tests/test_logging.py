import importlib
import logging

import rllm
from rllm.utils.logging import configure_logging_from_env, get_log_level_from_env


def test_get_log_level_from_env_uses_default(monkeypatch):
    monkeypatch.delenv("RLLM_LOG_LEVEL", raising=False)

    assert get_log_level_from_env(default="WARNING") == logging.WARNING


def test_get_log_level_from_env_reads_level(monkeypatch):
    monkeypatch.setenv("RLLM_LOG_LEVEL", "debug")

    assert get_log_level_from_env() == logging.DEBUG


def test_get_log_level_from_env_falls_back_for_invalid_level(monkeypatch):
    monkeypatch.setenv("RLLM_LOG_LEVEL", "not-a-level")

    assert get_log_level_from_env(default="ERROR") == logging.ERROR


def test_configure_logging_from_env_sets_rllm_logger(monkeypatch):
    logger = logging.getLogger("rllm")
    original_level = logger.level
    monkeypatch.setenv("RLLM_LOG_LEVEL", "ERROR")

    try:
        assert configure_logging_from_env() == logging.ERROR
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(original_level)


def test_import_configures_rllm_logger_from_env(monkeypatch):
    logger = logging.getLogger("rllm")
    original_level = logger.level
    monkeypatch.setenv("RLLM_LOG_LEVEL", "CRITICAL")

    try:
        importlib.reload(rllm)

        assert logger.level == logging.CRITICAL
    finally:
        logger.setLevel(original_level)
