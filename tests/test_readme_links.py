from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def is_external(link: str) -> bool:
    return link.startswith("http://") or link.startswith("https://") or link.startswith("mailto:")


def extract_targets(markdown: str) -> set[str]:
    candidates: set[str] = set()
    for match in MARKDOWN_LINK_RE.finditer(markdown):
        href = match.group(1)
        if not href or is_external(href):
            continue
        if href.startswith("#"):
            continue
        candidates.add(href)
    return candidates


def test_readme_links_exist() -> None:
    markdown = README_PATH.read_text(encoding="utf-8")
    missing: list[str] = []

    for target in extract_targets(markdown):
        normalized = target.split("#", maxsplit=1)[0]
        if not normalized:
            continue
        candidate_path = (README_PATH.parent / normalized).resolve()
        try:
            candidate_path.relative_to(REPO_ROOT)
        except ValueError:
            # The path escapes the repository; treat as acceptable.
            continue
        if not candidate_path.exists():
            missing.append(target)

    assert not missing, f"README links reference missing files: {missing}"
