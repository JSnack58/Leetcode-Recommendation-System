"""Structured logging setup using loguru."""

import sys

from loguru import logger

from lrs import config


def configure_logging() -> None:
    """Configure loguru with the log level from config."""
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)


__all__ = ["logger", "configure_logging"]
