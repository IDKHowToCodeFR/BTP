from __future__ import annotations

import pandas as pd
import pytest

from agentic_selection.evaluation.report import regret_table, render_markdown_tables


def make_stable_df(is_synthetic=False):
    rows = []
    for cond in ["global_fixed", "lookup_table", "agent_weights_only", "agent_full"]:
        for i in range(3):
            rows.append(
                {
                    "task_key": "streaming",
                    "condition": cond,
                    "regret": 0.1 * i,
                    "fallback_triggered": False,
                    "latency_seconds": 0.05,
                    "api_calls": 1 if "agent" in cond else 0,
                    "is_synthetic_data": is_synthetic,
                }
            )
    return pd.DataFrame(rows)


def test_regret_table_basic_shape():
    df = make_stable_df()
    table = regret_table(df)
    assert "streaming" in table.index
    assert "Global fixed" in table.columns


def test_synthetic_data_blocks_report_by_default():
    df = make_stable_df(is_synthetic=True)
    with pytest.raises(RuntimeError):
        regret_table(df, allow_synthetic=False)


def test_synthetic_data_allowed_with_explicit_override():
    df = make_stable_df(is_synthetic=True)
    table = regret_table(df, allow_synthetic=True)
    assert table.attrs.get("warning") is not None


def test_render_markdown_tables_includes_warning_banner_for_synthetic_data():
    df = make_stable_df(is_synthetic=True)
    md = render_markdown_tables(df, None, allow_synthetic=True)
    assert "SYNTHETIC" in md


def test_render_markdown_tables_no_warning_for_real_data():
    df = make_stable_df(is_synthetic=False)
    md = render_markdown_tables(df, None, allow_synthetic=False)
    assert "SYNTHETIC" not in md


def test_render_markdown_tables_includes_table_54_when_wsdream_df_given():
    df = make_stable_df(is_synthetic=False)
    wsdream_df = make_stable_df(is_synthetic=False)
    md = render_markdown_tables(df, None, allow_synthetic=False, wsdream_df=wsdream_df)
    assert "Table 5.4" in md
    assert "WS-DREAM" in md


def test_render_markdown_tables_omits_table_54_when_wsdream_df_not_given():
    df = make_stable_df(is_synthetic=False)
    md = render_markdown_tables(df, None, allow_synthetic=False)
    assert "Table 5.4" not in md
