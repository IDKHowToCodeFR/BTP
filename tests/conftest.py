from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS  # noqa: E402
from agentic_selection.data.synthetic import generate_synthetic_qws  # noqa: E402
from agentic_selection.data.preprocessing import normalize_benefit_oriented  # noqa: E402


@pytest.fixture
def small_2attr_pool() -> pd.DataFrame:
    """The hand-computable A/B/C/D example used throughout development:
    both weighted_sum and TOPSIS have exact closed-form expected values
    for this pool (see tests/test_weighted_sum.py, tests/test_topsis.py).
    """
    return pd.DataFrame(
        {"attr1": [0.8, 0.4, 0.6, 0.3], "attr2": [0.2, 0.9, 0.5, 0.3]},
        index=["A", "B", "C", "D"],
    )


@pytest.fixture
def synthetic_qws_raw() -> pd.DataFrame:
    return generate_synthetic_qws(n=200, seed=123)


@pytest.fixture
def synthetic_qws_normalized(synthetic_qws_raw) -> pd.DataFrame:
    return normalize_benefit_oriented(synthetic_qws_raw)


@pytest.fixture
def real_qws_path() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "qws" / "QWS_Dataset_v2.txt"
