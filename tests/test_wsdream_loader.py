from __future__ import annotations

import pytest

from agentic_selection.data.wsdream_loader import (
    aggregate_wsdream2_file,
    load_wsdream_dataset1,
    wsdream1_to_candidate_pool,
)


def _write_fixture(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    # 2 header lines + 3 users, matching the real file's structure
    (data_dir / "userlist.txt").write_text(
        "[User ID]\t[IP Address]\t[Country]\t[Continent]\t[AS]\t[Latitude]\t[Longitude]\t[Region]\t[City]\n"
        "====================\n"
        "0\t1.2.3.4\tUS\tNA\tAS1\t1.0\t2.0\tRegion1\tCity1\n"
        "1\t1.2.3.5\tUS\tNA\tAS1\t1.0\t2.0\tRegion1\tCity1\n"
        "2\t1.2.3.6\tUS\tNA\tAS1\t1.0\t2.0\tRegion1\tCity1\n"
    )
    (data_dir / "wslist.txt").write_text(
        "[Service ID]\t[WSDL Address]\t[Provider]\t[IP]\t[Country]\t[Continent]\t[AS]\t[Lat]\t[Lon]\t[Region]\t[City]\n"
        "====================\n"
        "0\thttp://a.wsdl\tProvA\t1.1.1.1\tUS\tNA\tAS1\t1.0\t2.0\tR\tC\n"
        "1\thttp://b.wsdl\tProvB\t1.1.1.2\tUS\tNA\tAS1\t1.0\t2.0\tR\tC\n"
        "2\thttp://c.wsdl\tProvC\t1.1.1.3\tUS\tNA\tAS1\t1.0\t2.0\tR\tC\n"
        "3\thttp://d.wsdl\tProvD\t1.1.1.4\tUS\tNA\tAS1\t1.0\t2.0\tR\tC\n"
    )
    # 3 users x 4 services, with a trailing tab (mirrors the real file's quirk)
    # and one -1 missing-value sentinel
    rt_lines = [
        "1.0\t2.0\t3.0\t-1\t\n",
        "1.5\t2.5\t3.5\t4.5\t\n",
        "1.2\t2.2\t3.2\t4.2\t\n",
    ]
    (data_dir / "rtMatrix.txt").write_text("".join(rt_lines))
    tp_lines = [
        "10.0\t20.0\t30.0\t40.0\t\n",
        "11.0\t-1\t31.0\t41.0\t\n",
        "12.0\t22.0\t32.0\t42.0\t\n",
    ]
    (data_dir / "tpMatrix.txt").write_text("".join(tp_lines))


def test_load_wsdream_dataset1_parses_shapes_correctly(tmp_path):
    data_dir = tmp_path / "wsdream1"
    _write_fixture(data_dir)
    with pytest.warns(UserWarning):  # shape (3,4) != documented (339, 5825)
        result = load_wsdream_dataset1(data_dir)
    assert result["rt"].shape == (3, 4)
    assert result["tp"].shape == (3, 4)
    assert len(result["users"]) == 3
    assert len(result["services"]) == 4


def test_load_wsdream_dataset1_missing_sentinel_converted_to_nan(tmp_path):
    data_dir = tmp_path / "wsdream1"
    _write_fixture(data_dir)
    with pytest.warns(UserWarning):
        result = load_wsdream_dataset1(data_dir)
    assert result["rt"].isnull().sum().sum() == 1  # the one -1 in rtMatrix
    assert result["tp"].isnull().sum().sum() == 1  # the one -1 in tpMatrix
    assert not (result["rt"].to_numpy() == -1).any()


def test_load_wsdream_dataset1_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wsdream_dataset1(tmp_path / "does_not_exist")


def test_wsdream1_to_candidate_pool_filters_by_min_observations(tmp_path):
    data_dir = tmp_path / "wsdream1"
    _write_fixture(data_dir)
    with pytest.warns(UserWarning):
        result = load_wsdream_dataset1(data_dir)
    # service col 3 in rt has 2/3 valid obs (one -1); require min_observations=3 -> excluded
    pool = wsdream1_to_candidate_pool(result, min_observations=3)
    service_ids = list(result["services"].index)
    assert service_ids[3] not in pool.index  # excluded: only 2 valid rt observations
    assert service_ids[0] in pool.index  # 3/3 valid


def test_wsdream1_to_candidate_pool_computes_correct_means(tmp_path):
    data_dir = tmp_path / "wsdream1"
    _write_fixture(data_dir)
    with pytest.warns(UserWarning):
        result = load_wsdream_dataset1(data_dir)
    pool = wsdream1_to_candidate_pool(result, min_observations=1)
    service_ids = list(result["services"].index)
    # service 0's rt values: 1.0, 1.5, 1.2 -> mean 1.2333...
    assert pool.loc[service_ids[0], "response_time"] == pytest.approx((1.0 + 1.5 + 1.2) / 3)


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/wsdream1/rtMatrix.txt").exists(),
    reason="real WS-DREAM Dataset #1 not downloaded (run scripts/01_download_data.py)",
)
def test_real_wsdream1_loads_and_aggregates_without_error():
    """Regression test for two bugs only reproducible on the real file:
    (1) userlist.txt/wslist.txt contain a handful of non-UTF-8 bytes (a
    mis-encoded character in a Brazilian ministry's name) that crash a
    plain pd.read_csv; (2) Series.combine(other, min) raised
    'truth value of a Series is ambiguous' against the real rt/tp
    observation-count Series shape, despite working on toy examples.
    Both are fixed in wsdream_loader.py; this test pins the fix against
    the actual file rather than only a small hand-written fixture.
    """
    from pathlib import Path

    result = load_wsdream_dataset1(Path("data/raw/wsdream1"))
    assert result["rt"].shape == (339, 5825)
    assert result["tp"].shape == (339, 5825)
    pool = wsdream1_to_candidate_pool(result, min_observations=30)
    assert len(pool) > 5000  # most of the 5825 services should clear this modest bar
    assert not pool[["response_time", "throughput"]].isnull().any().any()


def test_aggregate_wsdream2_file_computes_service_time_means(tmp_path):
    path = tmp_path / "rtdata.txt"
    path.write_text(
        "0 0 0 1.0\n"
        "1 0 0 3.0\n"
        "0 1 0 -1\n"
        "1 1 0 8.0\n"
        "0 0 1 5.0\n"
        "1 0 1 7.0\n"
        "0 1 1 9.0\n"
        "1 1 1 11.0\n"
    )

    means, counts = aggregate_wsdream2_file(
        path,
        "response_time",
        expected_users=2,
        expected_services=2,
        expected_timeslices=2,
        chunksize=3,
    )

    assert means.loc[0, 0] == pytest.approx(2.0)
    assert means.loc[0, 1] == pytest.approx(8.0)
    assert means.loc[1, 0] == pytest.approx(6.0)
    assert counts.loc[0, 1] == 1
    assert counts.loc[1, 1] == 2
