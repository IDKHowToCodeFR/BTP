from __future__ import annotations

import pytest

from agentic_selection.data.qws_loader import parse_qws_text, QWS_ALL_RAW_COLUMNS


SAMPLE_QWS_TEXT = """\
## QWS Dataset sample (fixture for testing, NOT the real 2507-row file)
## Columns: response_time,availability,throughput,successability,reliability,compliance,best_practices,latency,documentation,service_name,wsdl_address
##
107.0,63,15.1,63,73,100,84,4,2,NormalService,http://example.com/service1.wsdl
# a row with an embedded comma inside the WSDL address, mirroring the real
# QWS_Dataset_v2.txt file's two known irregular rows (see qws_loader.py docstring)
214.67,63,12.5,63,73,78,84,25.67,12,wsaTestService,http://fisheye5.cenqua.com/viewrep/~raw,r=1.2/test.wsdl
50.0,90,20.0,95,80,95,90,10,50,AnotherService,http://example.com/service3.wsdl
"""


def test_parse_qws_text_handles_embedded_comma_row():
    with pytest.warns(UserWarning):  # row count won't match the documented 2507
        df = parse_qws_text(SAMPLE_QWS_TEXT)
    assert len(df) == 3
    assert list(df.columns[:len(QWS_ALL_RAW_COLUMNS) + 1])[1:] == QWS_ALL_RAW_COLUMNS
    # the embedded-comma row must have its WSDL address intact, absorbing the extra comma
    row = df[df["service_name"] == "wsaTestService"].iloc[0]
    assert row["wsdl_address"] == "http://fisheye5.cenqua.com/viewrep/~raw,r=1.2/test.wsdl"
    assert row["response_time"] == pytest.approx(214.67)
    assert row["documentation"] == pytest.approx(12.0)


def test_parse_qws_text_skips_comment_and_blank_lines():
    with pytest.warns(UserWarning):
        df = parse_qws_text(SAMPLE_QWS_TEXT)
    assert (df["service_name"] != "").all()


def test_parse_qws_text_numeric_columns_are_numeric():
    from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
    import pandas as pd

    with pytest.warns(UserWarning):
        df = parse_qws_text(SAMPLE_QWS_TEXT)
    for col in QWS_ATTRIBUTE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_parse_qws_text_assigns_unique_service_ids():
    with pytest.warns(UserWarning):
        df = parse_qws_text(SAMPLE_QWS_TEXT)
    assert df["service_id"].is_unique
    # NOTE: deliberately compare as plain lists, not `df.index.equals(df["service_id"])`
    # -- pandas' Index.equals() compared directly against a same-named Series
    # does not reduce to elementwise value equality the way you'd expect;
    # wrapping in pd.Index(...) or comparing as lists are the reliable forms.
    assert list(df.index) == list(df["service_id"])


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/qws/QWS_Dataset_v2.txt").exists(),
    reason="real QWS dataset not downloaded (run scripts/01_download_data.py)",
)
def test_real_qws_file_parses_to_exactly_2507_rows_no_warning(real_qws_path):
    import warnings

    from agentic_selection.data.qws_loader import load_qws

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = load_qws(real_qws_path)
    assert len(df) == 2507
