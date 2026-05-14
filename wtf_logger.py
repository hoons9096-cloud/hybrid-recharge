"""
wtf_logger.py — Centralised logging configuration for hybrid-recharge.

All modules should import their logger via:

    from wtf_logger import get_logger
    logger = get_logger(__name__)

Log levels follow Python standard:
    DEBUG   — detailed diagnostic information
    INFO    — confirmation that things are working
    WARNING — something unexpected but recoverable
    ERROR   — a serious problem that prevented some function

In the Streamlit app, console logging is typically set to WARNING to avoid
cluttering stdout. When running from CLI (e.g., evaluation_runner.py),
DEBUG or INFO can be useful.

Usage
-----
    from wtf_logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded %d observations", n)
    logger.warning("sigma_ho fallback to default %.2f", DEFAULT_SIGMA_HO)
"""

import logging
import sys

# ── Module-level configuration ────────────────────────────
_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_DEFAULT_LEVEL = logging.WARNING  # change to INFO/DEBUG for verbose output

# One-time handler setup on the root 'wtf' logger
_root_logger = logging.getLogger("wtf")
if not _root_logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    _root_logger.addHandler(_handler)
    _root_logger.setLevel(_DEFAULT_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'wtf' namespace.

    Example: get_logger("core_sim_v27") → logger named 'wtf.core_sim_v27'
    """
    return logging.getLogger(f"wtf.{name}")


def set_level(level: int | str = logging.DEBUG) -> None:
    """Change the global WTF log level at runtime.

    Useful for CLI tools or debugging:
        from wtf_logger import set_level
        set_level("DEBUG")
    """
    _root_logger.setLevel(level)
