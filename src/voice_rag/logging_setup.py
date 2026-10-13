"""Logging configuration shared by the CLI and scripts."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger. ``--verbose`` shows DEBUG lines from this package only."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("voice_rag").setLevel(level)
    # Third-party libraries are chatty at DEBUG; keep them quiet even in verbose mode.
    for noisy in ("httpx", "httpcore", "urllib3", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
