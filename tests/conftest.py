from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REPO_KB = Path(__file__).resolve().parent.parent / "kb"


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """A writable copy of the real knowledge base so tests can edit files freely."""
    target = tmp_path / "kb"
    shutil.copytree(REPO_KB, target)
    return target


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    return tmp_path / ".index"
