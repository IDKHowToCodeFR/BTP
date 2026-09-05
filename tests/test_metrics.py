from __future__ import annotations

import pandas as pd
import pytest

from agentic_selection.evaluation.metrics import (
    regret,
    adaptation_lag,
    stability,
    fallback_trigger_rate,
    mean_confidence_interval,
    summarize_decision_cost,
)


def test_regret_zero_when_method_picks_reference_optimal():
    pool = pd.DataFrame({"a": [0.9, 0.5, 0.3], "b": [0.1, 0.5, 0.9]}, index=["X", "Y", "Z"])
    ref_w = {"a": 0.8, "b": 0.2}
    method_scores = pd.Series([0.9, 0.2, 0.1], index=["X", "Y", "Z"])  # picks X, the ref-optimal
    assert regret(pool, method_scores, ref_w, ["a", "b"]) == pytest.approx(0.0)


def test_regret_positive_when_method_picks_suboptimal():
    pool = pd.DataFrame({"a": [0.9, 0.5, 0.3], "b": [0.1, 0.5, 0.9]}, index=["X", "Y", "Z"])
    ref_w = {"a": 0.8, "b": 0.2}
    method_scores = pd.Series([0.1, 0.2, 0.9], index=["X", "Y", "Z"])  # picks Z
    r = regret(pool, method_scores, ref_w, ["a", "b"])
    assert r == pytest.approx(0.32, abs=1e-9)


def test_adaptation_lag_every_round_reevaluation():
    # indices 0-5 are 'A' (6 of them), index 6 onward is 'B' -> switch first
    # observed at round index 6, degrade_start_round=4 -> lag = 6-4 = 2
    calls = iter(["A", "A", "A", "A", "A", "A", "B", "B"])
    result = adaptation_lag([None] * 8, "A", degrade_start_round=4, decide_fn=lambda p: next(calls), reevaluation_period=None)
    assert result.lag_rounds == 2
    assert not result.censored


def test_adaptation_lag_periodic_reevaluation_holds_between_checkpoints():
    # checkpoints only at round 0 and round 5 (period=5); decide_fn called twice
    calls = iter(["A", "B"])
    result = adaptation_lag([None] * 10, "A", degrade_start_round=4, decide_fn=lambda p: next(calls), reevaluation_period=5)
    assert result.lag_rounds == 1  # active stays 'A' through round 4, becomes 'B' at round 5 -> lag=5-4=1


def test_adaptation_lag_censored_when_never_switches():
    result = adaptation_lag([None] * 6, "A", degrade_start_round=2, decide_fn=lambda p: "A", reevaluation_period=None)
    assert result.censored
    assert result.lag_rounds is None


def test_stability_all_same_is_zero():
    assert stability(["A", "A", "A"]) == 0.0


def test_stability_all_different_approaches_one():
    assert stability(["A", "B", "C", "D"]) == 0.75


def test_stability_empty_list_is_zero():
    assert stability([]) == 0.0


def test_fallback_trigger_rate_basic():
    assert fallback_trigger_rate([True, True, False, False]) == 0.5
    assert fallback_trigger_rate([]) == 0.0


def test_mean_confidence_interval_single_value():
    ci = mean_confidence_interval([5.0])
    assert ci.mean == 5.0
    assert ci.lower == ci.upper == 5.0


def test_mean_confidence_interval_empty():
    ci = mean_confidence_interval([])
    assert ci.n == 0


def test_mean_confidence_interval_reasonable_bounds():
    ci = mean_confidence_interval([1, 2, 3, 4, 5])
    assert ci.lower < ci.mean < ci.upper


def test_summarize_decision_cost():
    s = summarize_decision_cost([0.1, 0.2, 0.3], [1, 1, 2])
    assert s.mean_latency_seconds == pytest.approx(0.2)
    assert s.mean_api_calls == pytest.approx(4 / 3)
    assert s.n_decisions == 3
