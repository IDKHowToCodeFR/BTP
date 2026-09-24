from __future__ import annotations

import json

import pandas as pd
import pytest

from agentic_selection.agent.controller import AgentController
from agentic_selection.agent.llm_backends import MockBackend
from agentic_selection.baselines import TASK_LOOKUP_TABLE
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.evaluation.protocol import run_stable_protocol, run_drift_protocol, STABLE_RESULT_FIELDS, DRIFT_RESULT_FIELDS
from agentic_selection.evaluation.storage import CsvStorage
from agentic_selection.tasks import TASK_PROFILES, HELD_OUT_TASKS


def _smart_responder(system_prompt, user_prompt):
    text = user_prompt.lower()
    best_key, best_score = "streaming", -1
    for key, w in TASK_LOOKUP_TABLE.items():
        score = sum(1 for tok in key.split("_") if tok in text)
        if score > best_score:
            best_score, best_key = score, key
    w = TASK_LOOKUP_TABLE[best_key]
    return json.dumps({"weights": w, "strategy": "topsis", "justification": f"matched {best_key}"})


@pytest.fixture
def controller(tmp_path):
    backend = MockBackend(responder=_smart_responder)
    return AgentController(backend, QWS_ATTRIBUTE_COLUMNS, tmp_path / "mem.jsonl")


def test_run_stable_protocol_produces_expected_row_count(synthetic_qws_normalized, controller, tmp_path):
    out_csv = tmp_path / "stable.csv"
    storage = CsvStorage(out_csv, ["task_key", "pool_seed", "condition"], STABLE_RESULT_FIELDS)
    small_held_out = HELD_OUT_TASKS[:1]
    df = run_stable_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_pools=2, pool_size=10, held_out_tasks=small_held_out,
    )
    n_tasks = len(TASK_PROFILES) + len(small_held_out)
    expected_rows = n_tasks * 2 * 4  # n_pools x 4 conditions
    assert len(df) == expected_rows


def test_run_stable_protocol_is_resumable_no_duplicate_rows(synthetic_qws_normalized, controller, tmp_path):
    out_csv = tmp_path / "stable.csv"
    storage = CsvStorage(out_csv, ["task_key", "pool_seed", "condition"], STABLE_RESULT_FIELDS)
    small_held_out = HELD_OUT_TASKS[:1]
    df1 = run_stable_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_pools=2, pool_size=10, held_out_tasks=small_held_out,
    )
    # re-run with the SAME output file -- should not duplicate any trial
    df2 = run_stable_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_pools=2, pool_size=10, held_out_tasks=small_held_out,
    )
    assert len(df1) == len(df2)
    dupes = df2.duplicated(subset=["task_key", "pool_seed", "condition"]).sum()
    assert dupes == 0


def test_run_stable_protocol_extending_n_pools_only_adds_new_rows(synthetic_qws_normalized, controller, tmp_path):
    out_csv = tmp_path / "stable.csv"
    storage = CsvStorage(out_csv, ["task_key", "pool_seed", "condition"], STABLE_RESULT_FIELDS)
    small_held_out = HELD_OUT_TASKS[:1]
    df1 = run_stable_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_pools=2, pool_size=10, held_out_tasks=small_held_out,
    )
    df2 = run_stable_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_pools=4, pool_size=10, held_out_tasks=small_held_out,
    )
    assert len(df2) > len(df1)
    # original rows must still be present unchanged
    assert set(zip(df1.task_key, df1.pool_seed, df1.condition)).issubset(
        set(zip(df2.task_key, df2.pool_seed, df2.condition))
    )


def test_run_drift_protocol_produces_rows_for_each_condition(synthetic_qws_normalized, controller, tmp_path):
    out_csv = tmp_path / "drift.csv"
    storage = CsvStorage(out_csv, ["task_key", "trial_seed", "condition"], DRIFT_RESULT_FIELDS)
    df = run_drift_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_trials=1, pool_size=10, n_rounds=8, degrade_start_round=3,
        task_profiles=TASK_PROFILES[:2],
    )
    assert set(df["condition"].unique()) <= {"global_fixed", "lookup_table", "agent_weights_only", "agent_full"}
    assert len(df) > 0


def test_run_drift_protocol_resumable(synthetic_qws_normalized, controller, tmp_path):
    out_csv = tmp_path / "drift.csv"
    storage = CsvStorage(out_csv, ["task_key", "trial_seed", "condition"], DRIFT_RESULT_FIELDS)
    df1 = run_drift_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_trials=1, pool_size=10, n_rounds=8, degrade_start_round=3,
        task_profiles=TASK_PROFILES[:1],
    )
    df2 = run_drift_protocol(
        synthetic_qws_normalized, QWS_ATTRIBUTE_COLUMNS, storage, controller,
        n_trials=1, pool_size=10, n_rounds=8, degrade_start_round=3,
        task_profiles=TASK_PROFILES[:1],
    )
    assert len(df1) == len(df2)
