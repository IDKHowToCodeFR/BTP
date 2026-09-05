from __future__ import annotations

import pandas as pd
import pytest

from agentic_selection.drift.simulate import (
    find_common_top_choice,
    simulate_drift_sequence,
)


def make_pool():
    return pd.DataFrame(
        {"a": [0.9, 0.5, 0.3], "b": [0.8, 0.4, 0.2]},
        index=["X", "Y", "Z"],
    )


def test_find_common_top_choice_consensus():
    pool = make_pool()
    methods = {
        "m1": lambda p: p["a"],  # X is top under a
        "m2": lambda p: p["b"],  # X is also top under b
    }
    assert find_common_top_choice(pool, methods) == "X"


def test_find_common_top_choice_no_consensus_returns_none():
    pool = make_pool()
    methods = {
        "m1": lambda p: p["a"],  # X top
        "m2": lambda p: -p["a"],  # Z top (reversed)
    }
    assert find_common_top_choice(pool, methods) is None


def test_simulate_drift_sequence_sudden_profile():
    pool = make_pool()
    seq = simulate_drift_sequence(
        pool, ["a", "b"], target_service_id="X", degraded_attributes=["a"],
        n_rounds=10, degrade_start_round=4, profile="sudden", magnitude=0.5, seed=0,
    )
    assert len(seq.pools) == 10
    # before drift: unaffected
    assert seq.pools[3].loc["X", "a"] == pytest.approx(0.9)
    # at/after drift: reduced by exactly `magnitude` fraction
    assert seq.pools[4].loc["X", "a"] == pytest.approx(0.9 * 0.5)
    assert seq.pools[9].loc["X", "a"] == pytest.approx(0.9 * 0.5)
    # other services and other attributes untouched throughout
    assert seq.pools[9].loc["Y", "a"] == pytest.approx(0.5)
    assert seq.pools[9].loc["X", "b"] == pytest.approx(0.8)


def test_simulate_drift_sequence_gradual_profile_monotonically_declines():
    pool = make_pool()
    seq = simulate_drift_sequence(
        pool, ["a", "b"], target_service_id="X", degraded_attributes=["a"],
        n_rounds=10, degrade_start_round=4, profile="gradual", magnitude=0.8, seed=0,
    )
    values = [seq.pools[i].loc["X", "a"] for i in range(4, 10)]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    assert values[0] == pytest.approx(0.9)  # ramp starts at 0 reduction


def test_simulate_drift_sequence_sudden_then_recover():
    pool = make_pool()
    seq = simulate_drift_sequence(
        pool, ["a", "b"], target_service_id="X", degraded_attributes=["a"],
        n_rounds=12, degrade_start_round=4, profile="sudden_then_recover", magnitude=0.9, seed=0,
    )
    assert seq.pools[4].loc["X", "a"] < seq.pools[3].loc["X", "a"]  # drops
    assert seq.pools[-1].loc["X", "a"] == pytest.approx(seq.pools[0].loc["X", "a"])  # recovers to original


def test_simulate_drift_sequence_invalid_target_raises():
    pool = make_pool()
    with pytest.raises(KeyError):
        simulate_drift_sequence(pool, ["a", "b"], target_service_id="nonexistent", degraded_attributes=["a"], n_rounds=5, degrade_start_round=2)


def test_simulate_drift_sequence_invalid_attribute_raises():
    pool = make_pool()
    with pytest.raises(ValueError):
        simulate_drift_sequence(pool, ["a", "b"], target_service_id="X", degraded_attributes=["nonexistent"], n_rounds=5, degrade_start_round=2)


def test_simulate_drift_sequence_bad_degrade_start_round_raises():
    pool = make_pool()
    with pytest.raises(ValueError):
        simulate_drift_sequence(pool, ["a", "b"], target_service_id="X", degraded_attributes=["a"], n_rounds=5, degrade_start_round=0)
    with pytest.raises(ValueError):
        simulate_drift_sequence(pool, ["a", "b"], target_service_id="X", degraded_attributes=["a"], n_rounds=5, degrade_start_round=5)
