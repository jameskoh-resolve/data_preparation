import pytest
import os
import shutil
from pathlib import Path

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Fixture providing a temporary directory for caching."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    yield cache_dir
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
