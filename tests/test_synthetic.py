from __future__ import annotations

import pytest

from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.data.synthetic import generate_synthetic_qws, _QWS_REFERENCE_STATS


def test_generate_synthetic_qws_shape_and_flag():
    df = generate_synthetic_qws(n=500, seed=1)
    assert len(df) == 500
    assert (df["is_synthetic"] == True).all()  # noqa: E712
    for col in QWS_ATTRIBUTE_COLUMNS:
        assert col in df.columns


def test_generate_synthetic_qws_values_within_reference_bounds():
    df = generate_synthetic_qws(n=1000, seed=2)
    for col, (lo, hi, mean, std) in _QWS_REFERENCE_STATS.items():
        assert df[col].min() >= lo - 1e-6
        assert df[col].max() <= hi + 1e-6


def test_generate_synthetic_qws_reproducible():
    df1 = generate_synthetic_qws(n=50, seed=42)
    df2 = generate_synthetic_qws(n=50, seed=42)
    for col in QWS_ATTRIBUTE_COLUMNS:
        assert (df1[col].to_numpy() == df2[col].to_numpy()).all()


def test_generate_synthetic_qws_different_seeds_differ():
    df1 = generate_synthetic_qws(n=50, seed=1)
    df2 = generate_synthetic_qws(n=50, seed=2)
    assert not (df1["response_time"].to_numpy() == df2["response_time"].to_numpy()).all()


def test_generate_synthetic_qws_unique_service_ids():
    df = generate_synthetic_qws(n=100, seed=3)
    assert df["service_id"].is_unique
