"""Configuration loading boundary for the research system."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


@lru_cache(maxsize=8)
def _load_config(path: Path) -> dict[str, Any]:
    """Load repository configuration without inventing runtime defaults."""
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def load_config(path: Path = Path("configs/default.yaml")) -> dict[str, Any]:
    """Load configuration using an absolute path for cache correctness."""
    return _load_config(path.resolve())
