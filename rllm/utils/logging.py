import logging
import os


def get_log_level_from_env(env_var: str = "RLLM_LOG_LEVEL", default: str = "INFO") -> int:
    default_level = getattr(logging, default.upper(), logging.INFO)
    level_name = os.getenv(env_var, default).upper()
    return getattr(logging, level_name, default_level)


def configure_logging_from_env(
    env_var: str = "RLLM_LOG_LEVEL",
    default: str = "INFO",
) -> int:
    """Set the rLLM logger level from RLLM_LOG_LEVEL."""
    level = get_log_level_from_env(env_var, default)
    logging.getLogger("rllm").setLevel(level)
    return level


class DuplicateLoggingFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg in self.seen:
            return False
        self.seen.add(record.msg)
        return True
