"""Small utility helpers shared across the backend package."""

from typing import Optional

import git


def get_repository_commit_sha() -> Optional[str]:
    """Get current git commit SHA or None if repo not found."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return str(repo.head.object.hexsha)
    except Exception:  # pragma: no cover - defensive
        return None
