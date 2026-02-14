import logging
from rich.logging import RichHandler


def get_logger(
    name: str = "defaultLogger",
    level: int = logging.INFO,
    verbose: bool = False,
) -> logging.Logger:

    logger = logging.getLogger(name)

    # Remove existing handlers from this logger only
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    logger.setLevel(level)
    logger.propagate = False  # prevent double printing via root

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Always add file handler
    file_handler = logging.FileHandler(f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Only add RichHandler when verbose=True
    if verbose:
        rich_handler = RichHandler(rich_tracebacks=True, show_path=False)
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)

    return logger