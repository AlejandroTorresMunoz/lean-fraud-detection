"""Tiny YAML config loader. Keeps experiments declarative (see configs/base.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    """Load an experiment config from YAML into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
