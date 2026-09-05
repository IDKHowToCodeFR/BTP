from __future__ import annotations

import pytest

from agentic_selection.baselines.lookup_table import (
    GLOBAL_FIXED_WEIGHTS,
    TASK_LOOKUP_TABLE,
    get_lookup_weights,
    nearest_profile_key,
)
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS


def test_all_profile_weights_sum_to_one_and_cover_all_attributes():
    for key, w in TASK_LOOKUP_TABLE.items():
        assert set(w.keys()) == set(QWS_ATTRIBUTE_COLUMNS), f"profile {key} attribute mismatch"
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6), f"profile {key} doesn't sum to 1"
        assert all(v >= 0 for v in w.values()), f"profile {key} has negative weight"


def test_global_fixed_weights_uniform_and_sum_to_one():
    assert sum(GLOBAL_FIXED_WEIGHTS.values()) == pytest.approx(1.0)
    values = list(GLOBAL_FIXED_WEIGHTS.values())
    assert all(v == pytest.approx(values[0]) for v in values)


def test_get_lookup_weights_exact_key():
    w = get_lookup_weights("streaming")
    assert w == TASK_LOOKUP_TABLE["streaming"]


def test_get_lookup_weights_case_and_spacing_insensitive():
    w = get_lookup_weights("Financial Transaction")
    assert w == TASK_LOOKUP_TABLE["financial_transaction"]


def test_get_lookup_weights_raise_mode_on_miss():
    with pytest.raises(KeyError):
        get_lookup_weights("totally unrelated free text", fallback="raise")


def test_get_lookup_weights_global_fallback_mode():
    w = get_lookup_weights("something with no match at all xyz123", fallback="global")
    assert w == GLOBAL_FIXED_WEIGHTS


def test_nearest_profile_key_no_match_returns_none():
    assert nearest_profile_key("zzz_completely_unrelated_zzz", similarity_cutoff=0.9) is None


def test_no_cost_attribute_referenced_anywhere():
    """Regression test for the corrected task-profile design: QWS has no
    monetary 'cost' attribute, so no lookup-table entry should reference one."""
    for w in TASK_LOOKUP_TABLE.values():
        assert "cost" not in w
