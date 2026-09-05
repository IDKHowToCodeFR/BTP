from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_selection.data.preprocessing import (
    normalize_benefit_oriented,
    impute_missing,
    sample_candidate_pool,
    inject_missingness,
)


def test_normalize_cost_attribute_inverted():
    df = pd.DataFrame({"response_time": [100.0, 200.0, 300.0], "availability": [50.0, 75.0, 100.0]})
    norm = normalize_benefit_oriented(df, attribute_cols=["response_time", "availability"], cost_attributes={"response_time"})
    # lowest raw response_time (100) should map to the HIGHEST normalized value (1.0), since it's cost-type
    assert norm.loc[0, "response_time"] == pytest.approx(1.0)
    assert norm.loc[2, "response_time"] == pytest.approx(0.0)
    # availability is benefit-type: highest raw value maps to highest normalized value
    assert norm.loc[2, "availability"] == pytest.approx(1.0)
    assert norm.loc[0, "availability"] == pytest.approx(0.0)


def test_normalize_all_values_within_unit_interval(synthetic_qws_raw):
    from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS

    norm = normalize_benefit_oriented(synthetic_qws_raw)
    for col in QWS_ATTRIBUTE_COLUMNS:
        assert norm[col].min() >= -1e-9
        assert norm[col].max() <= 1 + 1e-9


def test_normalize_degenerate_constant_column_becomes_neutral():
    df = pd.DataFrame({"const": [5.0, 5.0, 5.0], "other": [1.0, 2.0, 3.0]})
    with pytest.warns(UserWarning):
        norm = normalize_benefit_oriented(df, attribute_cols=["const", "other"], cost_attributes=set())
    assert (norm["const"] == 0.5).all()


def test_normalize_missing_attribute_column_raises():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(KeyError):
        normalize_benefit_oriented(df, attribute_cols=["a", "b"], cost_attributes=set())


def test_impute_missing_column_mean():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out = impute_missing(df, attribute_cols=["a"], strategy="column_mean")
    assert out["a"].iloc[1] == pytest.approx(2.0)
    assert not out["a"].isnull().any()


def test_sample_candidate_pool_reproducible(synthetic_qws_normalized):
    p1 = sample_candidate_pool(synthetic_qws_normalized, n=10, seed=5)
    p2 = sample_candidate_pool(synthetic_qws_normalized, n=10, seed=5)
    assert list(p1.index) == list(p2.index)


def test_sample_candidate_pool_different_seeds_differ(synthetic_qws_normalized):
    p1 = sample_candidate_pool(synthetic_qws_normalized, n=10, seed=1)
    p2 = sample_candidate_pool(synthetic_qws_normalized, n=10, seed=2)
    assert list(p1.index) != list(p2.index)


def test_sample_candidate_pool_too_large_raises(synthetic_qws_normalized):
    with pytest.raises(ValueError):
        sample_candidate_pool(synthetic_qws_normalized, n=len(synthetic_qws_normalized) + 1, seed=0)


def test_inject_missingness_reproducible_and_bounded():
    df = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float)})
    out = inject_missingness(df, ["a", "b"], fraction=0.2, seed=0)
    frac_missing = out[["a", "b"]].isnull().mean().mean()
    assert 0.1 < frac_missing < 0.3  # roughly 20%, allow sampling noise
