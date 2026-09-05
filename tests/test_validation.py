from __future__ import annotations

import pytest

from agentic_selection.agent.validation import (
    validate_agent_output,
    DEGENERATE_SINGLE_ATTR_THRESHOLD,
    default_degenerate_threshold,
)

ATTRS = ["a", "b", "c"]


def test_valid_response_passes_through_normalized():
    parsed = {"weights": {"a": 1.0, "b": 1.0, "c": 2.0}, "strategy": "topsis", "justification": "reasoned"}
    result = validate_agent_output(parsed, "some task", ATTRS)
    assert not result.fallback_triggered
    assert result.weights == {"a": 0.25, "b": 0.25, "c": 0.5}
    assert result.strategy == "topsis"


def test_none_parsed_triggers_fallback():
    result = validate_agent_output(None, "streaming task", ATTRS)
    assert result.fallback_triggered
    assert result.fallback_reason is not None


def test_missing_strategy_triggers_fallback():
    parsed = {"weights": {"a": 1.0, "b": 1.0, "c": 1.0}, "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert result.fallback_triggered


def test_strategy_not_in_menu_triggers_fallback():
    parsed = {"weights": {"a": 1.0, "b": 1.0, "c": 1.0}, "strategy": "made_up_strategy", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert result.fallback_triggered


def test_unknown_attribute_key_triggers_fallback():
    parsed = {"weights": {"a": 1.0, "price": 1.0}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert result.fallback_triggered
    assert "price" in result.fallback_reason


def test_negative_weight_triggers_fallback():
    parsed = {"weights": {"a": -0.5, "b": 1.0, "c": 0.5}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert result.fallback_triggered


def test_all_zero_weights_triggers_fallback():
    parsed = {"weights": {"a": 0.0, "b": 0.0, "c": 0.0}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert result.fallback_triggered


def test_degenerate_single_attribute_triggers_fallback():
    parsed = {"weights": {"a": 0.99, "b": 0.005, "c": 0.005}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert result.fallback_triggered
    assert "degenerate" in result.fallback_reason


def test_concentration_just_under_threshold_does_not_trigger_fallback():
    # ATTRS has 3 attributes -> default_degenerate_threshold(3) == 0.90; 0.80 should be fine.
    parsed = {"weights": {"a": 0.80, "b": 0.10, "c": 0.10}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert not result.fallback_triggered


def test_non_numeric_weight_value_triggers_fallback():
    parsed = {"weights": {"a": "high", "b": 1.0, "c": 1.0}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert result.fallback_triggered


def test_missing_justification_gets_placeholder_not_a_failure():
    parsed = {"weights": {"a": 1.0, "b": 1.0, "c": 1.0}, "strategy": "topsis"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert not result.fallback_triggered
    assert isinstance(result.justification, str) and len(result.justification) > 0


def test_fallback_weights_still_sum_to_one_and_cover_all_attrs():
    result = validate_agent_output(None, "streaming video delivery", ATTRS)
    assert set(result.weights.keys()) == set(ATTRS)
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_partial_weights_omission_allowed_if_not_degenerate():
    # omitting 'c' entirely is fine as long as the remaining mass isn't degenerate
    parsed = {"weights": {"a": 0.5, "b": 0.5}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", ATTRS)
    assert not result.fallback_triggered
    assert result.weights["c"] == 0.0


def test_default_degenerate_threshold_scales_with_attribute_count():
    assert default_degenerate_threshold(2) == 0.95
    assert default_degenerate_threshold(3) == 0.90
    assert default_degenerate_threshold(5) == 0.90
    assert default_degenerate_threshold(6) == 0.85
    assert default_degenerate_threshold(9) == 0.85


def test_two_attribute_schema_allows_lopsided_but_legitimate_weighting():
    # 90/10 on 2 attributes is ordinary (e.g. WS-DREAM's response_time/
    # throughput schema), NOT a hallucination sign -- must not fall back.
    two_attrs = ["response_time", "throughput"]
    parsed = {"weights": {"response_time": 0.10, "throughput": 0.90}, "strategy": "topsis", "justification": "throughput-dominant task"}
    result = validate_agent_output(parsed, "task", two_attrs)
    assert not result.fallback_triggered


def test_two_attribute_schema_still_catches_genuine_degeneracy():
    # 99/1 on 2 attributes is still implausible and should still fall back.
    two_attrs = ["response_time", "throughput"]
    parsed = {"weights": {"response_time": 0.99, "throughput": 0.01}, "strategy": "topsis", "justification": "x"}
    result = validate_agent_output(parsed, "task", two_attrs)
    assert result.fallback_triggered


def test_explicit_degenerate_threshold_override():
    parsed = {"weights": {"a": 0.80, "b": 0.10, "c": 0.10}, "strategy": "topsis", "justification": "x"}
    # 0.80 doesn't trip the default (0.90 for 3 attrs), but does trip an explicit stricter override
    result = validate_agent_output(parsed, "task", ATTRS, degenerate_threshold=0.75)
    assert result.fallback_triggered
