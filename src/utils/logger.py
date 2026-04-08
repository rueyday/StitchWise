"""Utility logging module"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger as _logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Configure logger for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
    """
    # Remove default handler
    _logger.remove()

    # Add console handler
    _logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )

    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            retention="7 days"
        )


def get_logger(name: str = __name__):
    """Get logger instance"""
    return _logger.bind(name=name)
