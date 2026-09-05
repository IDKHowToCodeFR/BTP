from __future__ import annotations

import pytest

from agentic_selection.baselines.wsdream_lookup_table import (
    WSDREAM_ATTRIBUTE_COLUMNS,
    WSDREAM_TASK_LOOKUP_TABLE,
    WSDREAM_GLOBAL_FIXED_WEIGHTS,
    get_wsdream_lookup_weights,
)
from agentic_selection.agent.validation import default_degenerate_threshold
from agentic_selection.tasks.wsdream_profiles import WSDREAM_TASK_PROFILES, WSDREAM_TASK_PROFILE_BY_KEY
from agentic_selection.tasks.profiles import TASK_PROFILE_BY_KEY


def test_wsdream_attribute_columns_is_exactly_two():
    assert WSDREAM_ATTRIBUTE_COLUMNS == ["response_time", "throughput"]


def test_all_wsdream_profile_weights_sum_to_one_and_cover_both_attributes():
    for key, w in WSDREAM_TASK_LOOKUP_TABLE.items():
        assert set(w.keys()) == set(WSDREAM_ATTRIBUTE_COLUMNS)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_wsdream_global_fixed_weights_uniform():
    assert WSDREAM_GLOBAL_FIXED_WEIGHTS == {"response_time": 0.5, "throughput": 0.5}


def test_wsdream_weights_stay_under_the_two_attribute_degeneracy_threshold():
    """The 90/10 splits are intentionally aggressive (see module
    docstring) but must stay under default_degenerate_threshold(2)=0.95,
    or the agent's own validation layer would flag its own lookup-table
    fallback as degenerate, which would be a self-contradiction."""
    threshold = default_degenerate_threshold(len(WSDREAM_ATTRIBUTE_COLUMNS))
    for w in WSDREAM_TASK_LOOKUP_TABLE.values():
        assert max(w.values()) < threshold


def test_streaming_and_iot_profiles_are_opposite_dominant_attribute():
    streaming = WSDREAM_TASK_LOOKUP_TABLE["streaming"]
    iot = WSDREAM_TASK_LOOKUP_TABLE["iot_telemetry_ingestion"]
    assert streaming["throughput"] > streaming["response_time"]
    assert iot["response_time"] > iot["throughput"]


def test_get_wsdream_lookup_weights_exact_and_fallback():
    assert get_wsdream_lookup_weights("streaming") == WSDREAM_TASK_LOOKUP_TABLE["streaming"]
    assert get_wsdream_lookup_weights("totally unrelated", fallback="global") == WSDREAM_GLOBAL_FIXED_WEIGHTS
    with pytest.raises(KeyError):
        get_wsdream_lookup_weights("totally unrelated", fallback="raise")


def test_wsdream_task_profiles_reuse_exact_description_text_from_main_profiles():
    """Deliberate design choice (see wsdream_lookup_table.py docstring):
    the WS-DREAM validation must show the LLM the exact same sentence as
    the QWS run, to test generalization across attribute *schemas*, not
    accidentally test sensitivity to reworded prompts instead."""
    for p in WSDREAM_TASK_PROFILES:
        assert p.description == TASK_PROFILE_BY_KEY[p.key].description


def test_wsdream_task_profile_dominant_attributes_are_within_schema():
    for p in WSDREAM_TASK_PROFILES:
        assert set(p.dominant_attributes) <= set(WSDREAM_ATTRIBUTE_COLUMNS)


def test_wsdream_task_profile_by_key_lookup_consistent():
    for p in WSDREAM_TASK_PROFILES:
        assert WSDREAM_TASK_PROFILE_BY_KEY[p.key] is p
