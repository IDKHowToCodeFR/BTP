"""Loads config.yaml (or a path passed via --config) into a plain dict.
Kept intentionally dumb (no schema/validation framework) -- this project
has one config file with a handful of fields, consumed directly by the
scripts in scripts/, and a validation framework would be more machinery
than the problem calls for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path | str = "config.yaml") -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Copy config.yaml from the "
            f"project root, or pass --config path/to/your_config.yaml."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
