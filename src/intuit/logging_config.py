"""
Centralized logging configuration for Intuit.
"""

import logging
import sys
import os
from typing import List


# Create a NullHandler that does nothing with log records
class NullHandler(logging.Handler):
    def emit(self, record):
        pass


# Create a custom filter that blocks all log messages except ERROR and CRITICAL
class SuppressFilter(logging.Filter):
    def __init__(self, level=logging.ERROR):
        self.level = level

    def filter(self, record):
        return record.levelno >= self.level


# Global instances
null_handler = NullHandler()
suppress_filter = SuppressFilter(logging.ERROR)

# Known loggers that need to be silenced
ALL_LOGGERS = [
    "mcp",
    "intuit",
    "chromadb",
    "urllib3",
    "httpx",
    "httpcore",
    "asyncio",
    "chromadb.telemetry",
    "chromadb.api",
    "chromadb.segment",
    "chromadb.db",
    "chromadb.server",
    "chromadb.config",
    "hnswlib",
]

# Store original basicConfig for restoration if needed
original_basicConfig = logging.basicConfig


def patched_basicConfig(**kwargs):
    """Patched version of basicConfig that does nothing to prevent other libraries from changing the config."""
    pass


def configure_logging(verbose: int = 0, quiet: bool = True) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose: Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3+=DEBUG)
        quiet: If True, apply aggressive silencing for non-verbose mode
    """
    global suppress_filter

    # Set log levels based on verbosity count
    if verbose == 0:
        level = logging.ERROR
    elif verbose == 1:
        level = logging.WARNING
    elif verbose == 2:
        level = logging.INFO
    elif verbose >= 3:
        level = logging.DEBUG
    else:
        level = logging.ERROR

    # Configure basic logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Update the filter level
    suppress_filter.level = level

    # Apply to root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(null_handler)
    root_logger.addFilter(suppress_filter)

    # Remove all existing handlers from the root logger except our null handler
    for handler in list(root_logger.handlers):
        if handler != null_handler:
            root_logger.removeHandler(handler)

    # Apply to all known loggers
    for logger_name in ALL_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(null_handler)
        logger.addFilter(suppress_filter)
        logger.propagate = False  # Prevent propagation to parent loggers

    # Monkey patch the logging module to prevent other libraries from changing the configuration
    if quiet and verbose == 0:
        logging.basicConfig = patched_basicConfig


def aggressive_silence() -> None:
    """
    Apply aggressive silencing for environments with excessive logs.
    This is useful for testing or when you want minimal output.
    """
    configure_logging(verbose=0, quiet=True)

    # Additional silencing for specific scenarios
    for logger_name in [
        "",
        "mcp",
        "intuit",
        "chromadb",
        "urllib3",
        "httpx",
        "httpcore",
        "asyncio",
        "chromadb.telemetry",
        "chromadb.api",
        "chromadb.segment",
        "chromadb.db",
        "chromadb.server",
        "chromadb.config",
        "hnswlib",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.addHandler(null_handler)
        logger.addFilter(suppress_filter)
        logger.propagate = False


def silence_for_agent_creation() -> None:
    """
    Special silencing configuration for agent creation to avoid noise.
    """
    if not sys.stdin.isatty():
        # Redirect stderr to /dev/null for non-interactive mode
        old_stderr = sys.stderr
        try:
            with open(os.devnull, "w") as devnull:
                sys.stderr = devnull
                aggressive_silence()
        finally:
            sys.stderr = old_stderr
    else:
        aggressive_silence()


def restore_logging() -> None:
    """
    Restore original logging configuration if needed.
    """
    logging.basicConfig = original_basicConfig
