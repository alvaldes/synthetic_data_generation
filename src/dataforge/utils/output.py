"""
Output path utilities for timestamped file naming.

Provides helpers to append timestamps to filenames so that pipeline
runs produce unique output files instead of overwriting previous ones.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


def timestamped_filename(
    directory: str,
    base_name: str,
    suffix: str = ".csv",
    timestamp: Optional[str] = None,
) -> str:
    """
    Return ``<directory>/<base_name>_<timestamp><suffix>``.

    When *timestamp* is ``None`` (the default), the current local time is
    used in ``YYYYMMDD_HHMMSS`` format so that multiple runs within the
    same second are still possible (though unlikely).

    Parameters
    ----------
    directory : str
        Output directory path.
    base_name : str
        File name without extension (e.g. ``"salony_tasks"``).
    suffix : str
        File extension including the dot (default ``.csv``).
    timestamp : str, optional
        Explicit timestamp string; auto-generated when ``None``.

    Returns
    -------
    str
        Absolute or relative path string.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(directory) / f"{base_name}_{ts}{suffix}")
