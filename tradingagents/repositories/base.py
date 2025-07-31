"""
Base repository class with common utilities.
"""

from pathlib import Path


class BaseRepository:
    """Base repository class with shared utility methods."""

    def _ensure_path_exists(self, path: Path) -> None:
        """
        Ensure a directory path exists, creating it if necessary.

        Args:
            path: Path to ensure exists (can be file path - will create parent dirs)
        """
        if path.suffix:  # It's a file path, create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory path, create the directory itself
            path.mkdir(parents=True, exist_ok=True)
