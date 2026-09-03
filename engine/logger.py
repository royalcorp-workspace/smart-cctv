"""Centralized persistent logging engine with RotatingFileHandler and console output."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_WORKSPACE_DIR: Path = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR: Path = _WORKSPACE_DIR / "logs"
DEFAULT_LOG_FILE: Path = DEFAULT_LOG_DIR / "app.log"
LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "smart_cctv",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a thread-safe logger with console and rotating file handlers."""
    target_logger = logging.getLogger(name)
    target_logger.setLevel(level)

    # Prevent handler duplication if already configured
    if target_logger.hasHandlers():
        return target_logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    target_logger.addHandler(console_handler)

    # 2. Rotating File Handler (5 MB per file, max 3 backups)
    target_path = log_file if log_file is not None else DEFAULT_LOG_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(target_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    target_logger.addHandler(file_handler)

    return target_logger


# Default system-wide logger instance
logger: logging.Logger = setup_logger()
