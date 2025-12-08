import logging
from rich.logging import RichHandler


def get_logger(
    name: str = "defaultLogger", level: int = logging.INFO
) -> logging.Logger:
    """
    Returns a logger configured with RichHandler, ensuring that
    any existing handlers on this logger are removed first.

    :param name: Name of the logger.
    :param level: Logging level.
    :return: Configured logger.
    """
    logger = logging.getLogger(name)

    # Remove any existing handlers from this logger
    # (the loop approach or just `logger.handlers = []`)
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Remove any existing handlers from the root logger
    root_logger = logging.getLogger()
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    # Set the logging level
    logger.setLevel(level)

    # If logging is globally disabled, do not add any handlers
    if not logging.getLogger().isEnabledFor(logging.CRITICAL):
        return logger

    # Create a RichHandler
    rich_handler = RichHandler(rich_tracebacks=True, show_path=False)

    # Set the format for the logger
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)

    # Add the RichHandler to the logger
    logger.addHandler(rich_handler)

    return logger
