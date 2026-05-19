"""Structured logging setup using loguru."""

from loguru import logger
import sys

from lrs import config


def configure_logging() -> None:
    """Configure loguru with the log level from config."""
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)


def get_logger(name: str | None = None):
    """Return loguru logger (name ignored, kept for API compatibility)."""
    return logger


__all__ = ["logger", "configure_logging", "get_logger"]
