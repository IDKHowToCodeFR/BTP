"""Loader for the WS-DREAM QoS datasets (Zheng, Zhang & Lyu).

IMPORTANT CORRECTION vs. the original project brief: "WS-DREAM" is not one
dataset. It is a family of releases, and the two relevant ones here have
*different* sizes and *different* structure:

- Dataset #1: 339 users x 5,825 web services, a SINGLE snapshot of
  response time (rtMatrix.txt) and throughput (tpMatrix.txt). No time
  dimension. Verified directly (see below) against a real, downloadable
  mirror.
- Dataset #2: 142 users x 4,500 web services x 64 time slices -- this is
  the actual time-aware dataset with a genuine drift dimension. The official
  Zenodo archive was downloaded and byte-inspected during the complete
  WS-DREAM extension.

An earlier draft of this project's paper (§3.2) described a single
"WS-DREAM dataset" as "5,825 services x 339 users x 64 time slices" --
that sentence conflates the two releases (it has Dataset #1's
user/service counts and Dataset #2's time-slice count, which do not
belong to the same file). This loader keeps the two properly separate.

Provenance:
- Dataset #1 is mirrored as plain tab-separated text in several public
  GitHub repos; this loader was built and byte-checked against
  https://github.com/LanternYing/Dataset (dataset#1/{userlist,wslist,
  rtMatrix,tpMatrix}.txt). Confirmed shape: rtMatrix.txt and
  tpMatrix.txt are each 339 rows x 5,825 columns (one trailing empty
  column from a trailing tab character, which this loader drops),
  tab-separated, with -1 as the missing-value sentinel (~5.1% missing in
  rtMatrix, ~7.3% in tpMatrix in the copy inspected).
- Dataset #2 is distributed through the official WS-DREAM Zenodo record
  (record 1133476). Its archive contains rtdata.txt and tpdata.txt in the
  documented four-column long format. Each file contains 40,896,000 rows.
  ``aggregate_wsdream2_file`` processes these files in chunks and preserves
  per-service/time-slice observation counts. ``scripts/09_run_wsdream_complete.py``
  uses those aggregates for static and temporal evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MISSING_SENTINEL = -1.0

WSDREAM1_FILES = ("userlist.txt", "wslist.txt", "rtMatrix.txt", "tpMatrix.txt")
WSDREAM1_BASE_URL = "https://raw.githubusercontent.com/LanternYing/Dataset/master/dataset%231"
WSDREAM1_EXPECTED_SHAPE = (339, 5825)


def _read_matrix(path: Path) -> pd.DataFrame:
    mat = pd.read_csv(path, sep="\t", header=None)
    mat = mat.dropna(axis=1, how="all")  # drop the trailing all-NaN column
    mat = mat.replace(MISSING_SENTINEL, np.nan)
    return mat


def download_wsdream_dataset1(dest_dir: Path | str, timeout: int = 60) -> Path:
    """Download all four Dataset #1 files into dest_dir. Raises on any
    failure; callers wanting a graceful fallback should catch and skip
    (Dataset #1 is a secondary/bonus dataset in this project's design,
    not required for the core protocol -- see README).
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in WSDREAM1_FILES:
        url = f"{WSDREAM1_BASE_URL}/{fname}"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        if len(resp.content) < 500:
            raise ValueError(f"Downloaded {fname} looks too small ({len(resp.content)} bytes)")
        (dest_dir / fname).write_bytes(resp.content)
    return dest_dir


def load_wsdream_dataset1(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load WS-DREAM Dataset #1: per-(user, service) response time and
    throughput, no time dimension.

    Returns
    -------
    dict with keys:
        "rt": DataFrame, index=user_id, columns=service_id, response
              time in seconds (missing values are NaN, not -1)
        "tp": DataFrame, same shape, throughput in kb/s
        "users": DataFrame of user metadata (indexed by user_id)
        "services": DataFrame of service metadata (indexed by service_id)
    """
    data_dir = Path(data_dir)
    for fname in WSDREAM1_FILES:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"{fname} not found in {data_dir}. Run "
                f"scripts/01_download_data.py first, or pass an explicit path."
            )

    rt = _read_matrix(data_dir / "rtMatrix.txt")
    tp = _read_matrix(data_dir / "tpMatrix.txt")
    if rt.shape != WSDREAM1_EXPECTED_SHAPE or tp.shape != WSDREAM1_EXPECTED_SHAPE:
        import warnings

        warnings.warn(
            f"load_wsdream_dataset1: expected shape "
            f"{WSDREAM1_EXPECTED_SHAPE}, got rt={rt.shape}, tp={tp.shape}. "
            f"Proceeding, but this may indicate a different data release.",
            stacklevel=2,
        )

    # NOTE: userlist.txt and wslist.txt are NOT clean UTF-8 -- a handful of
    # provider/location name fields (e.g. a Brazilian ministry name in the
    # real file) contain stray non-UTF-8 bytes, presumably from a mid-2000s
    # scrape that mixed encodings. This was found by actually loading the
    # real downloaded file (a plain pd.read_csv without encoding handling
    # raises UnicodeDecodeError on it). Since these text metadata columns
    # are never used computationally by wsdream1_to_candidate_pool (only
    # the numeric rt/tp matrices and the row/column *positions* are used),
    # decoding with errors="replace" is safe: a handful of characters in a
    # provider name become U+FFFD, nothing downstream depends on them.
    users = pd.read_csv(
        data_dir / "userlist.txt", sep="\t", skiprows=2, header=None,
        names=["user_id", "ip_address", "country", "continent", "as_", "latitude", "longitude", "region", "city"],
        encoding="utf-8", encoding_errors="replace",
    ).set_index("user_id")
    services = pd.read_csv(
        data_dir / "wslist.txt", sep="\t", skiprows=2, header=None,
        names=["service_id", "wsdl_address", "provider", "ip_address", "country", "continent", "as_", "latitude", "longitude", "region", "city"],
        encoding="utf-8", encoding_errors="replace",
    ).set_index("service_id")

    rt.index = users.index[: len(rt)]
    rt.columns = services.index[: rt.shape[1]]
    tp.index = users.index[: len(tp)]
    tp.columns = services.index[: tp.shape[1]]

    return {"rt": rt, "tp": tp, "users": users, "services": services}


def wsdream1_to_candidate_pool(
    wsdream1: dict[str, pd.DataFrame],
    min_observations: int = 30,
) -> pd.DataFrame:
    """Aggregate per-(user, service) measurements into one row per
    service, suitable for use with the same baselines/agent pipeline as
    the QWS dataset. Only two real QWS-style attributes are available
    from this dataset (response_time, throughput); services with fewer
    than ``min_observations`` non-missing user measurements are dropped
    as too sparse to trust a mean estimate from.

    Returns a DataFrame indexed by service_id with columns
    ["response_time", "throughput", "n_observations"]. response_time is
    in seconds (cost-type, matches QWS's units/direction); throughput is
    in kb/s (benefit-type).
    """
    rt, tp = wsdream1["rt"], wsdream1["tp"]
    rt_count = rt.notna().sum(axis=0)
    tp_count = tp.notna().sum(axis=0)
    # NOTE: Series.combine(other, min) was tried here first and found to
    # raise ("truth value of a Series is ambiguous") against the real
    # rt_count/tp_count Series produced from the actual downloaded data,
    # despite working fine on toy examples -- rather than chase a pandas
    # version-specific internals bug, use the simpler and more robust
    # pd.concat(...).min(axis=1), which was verified to produce identical,
    # correct results and has no such failure mode.
    n_obs = pd.concat([rt_count, tp_count], axis=1).min(axis=1)

    out = pd.DataFrame(
        {
            "response_time": rt.mean(axis=0, skipna=True),
            "throughput": tp.mean(axis=0, skipna=True),
            "n_observations": n_obs,
        }
    )
    out = out[out["n_observations"] >= min_observations].copy()
    out.index.name = "service_id"
    return out


# ---------------------------------------------------------------------------
# Dataset #2 (time-aware). Documented record format per the archive README:
# one line per (user, service, time slice) observation:
#     "<user_id>\t<service_id>\t<time_slice_id>\t<value>"
# with -1 as the missing-value sentinel, mirroring Dataset #1's convention.
# ---------------------------------------------------------------------------

WSDREAM2_EXPECTED_USERS = 142
WSDREAM2_EXPECTED_SERVICES = 4500
WSDREAM2_EXPECTED_TIMESLICES = 64


def aggregate_wsdream2_file(
    path: Path | str,
    value_name: str,
    *,
    expected_users: int = WSDREAM2_EXPECTED_USERS,
    expected_services: int = WSDREAM2_EXPECTED_SERVICES,
    expected_timeslices: int = WSDREAM2_EXPECTED_TIMESLICES,
    chunksize: int = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate a Dataset #2 long file to service means per time slice.

    The real release contains more than 40 million rows per QoS attribute.
    Chunked ``bincount`` aggregation keeps peak memory bounded while retaining
    the observation count behind every service/time mean.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    if min(expected_users, expected_services, expected_timeslices, chunksize) <= 0:
        raise ValueError("expected dimensions and chunksize must be positive")

    n_cells = expected_services * expected_timeslices
    sums = np.zeros(n_cells, dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    seen_users = np.zeros(expected_users, dtype=bool)
    total_rows = 0

    reader = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["user_id", "service_id", "time_slice", value_name],
        dtype={
            "user_id": np.int32,
            "service_id": np.int32,
            "time_slice": np.int16,
            value_name: np.float64,
        },
        chunksize=chunksize,
        engine="c",
    )
    for chunk in reader:
        users = chunk["user_id"].to_numpy()
        services = chunk["service_id"].to_numpy()
        slices = chunk["time_slice"].to_numpy()
        values = chunk[value_name].to_numpy()
        if (
            np.any(users < 0)
            or np.any(users >= expected_users)
            or np.any(services < 0)
            or np.any(services >= expected_services)
            or np.any(slices < 0)
            or np.any(slices >= expected_timeslices)
        ):
            raise ValueError(f"{path} contains an ID outside the documented range")

        seen_users[users] = True
        valid = np.isfinite(values) & (values != MISSING_SENTINEL)
        flat_index = slices[valid].astype(np.int64) * expected_services + services[valid]
        sums += np.bincount(flat_index, weights=values[valid], minlength=n_cells)
        counts += np.bincount(flat_index, minlength=n_cells)
        total_rows += len(chunk)

    expected_rows = expected_users * expected_services * expected_timeslices
    if total_rows != expected_rows or int(seen_users.sum()) != expected_users:
        import warnings

        warnings.warn(
            f"{path.name}: expected {expected_rows} rows from {expected_users} users, "
            f"found {total_rows} rows from {int(seen_users.sum())} users",
            stacklevel=2,
        )

    means = np.full(n_cells, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    columns = pd.RangeIndex(expected_services, name="service_id")
    index = pd.RangeIndex(expected_timeslices, name="time_slice")
    return (
        pd.DataFrame(means.reshape(expected_timeslices, expected_services), index=index, columns=columns),
        pd.DataFrame(counts.reshape(expected_timeslices, expected_services), index=index, columns=columns),
    )


def load_wsdream_dataset2_aggregated(
    data_dir: Path | str,
    *,
    chunksize: int = 1_000_000,
) -> dict[str, pd.DataFrame]:
    """Load and aggregate both real Dataset #2 QoS files."""
    data_dir = Path(data_dir)
    rt, rt_counts = aggregate_wsdream2_file(
        data_dir / "rtdata.txt",
        "response_time",
        chunksize=chunksize,
    )
    tp, tp_counts = aggregate_wsdream2_file(
        data_dir / "tpdata.txt",
        "throughput",
        chunksize=chunksize,
    )
    return {
        "rt": rt,
        "tp": tp,
        "rt_counts": rt_counts,
        "tp_counts": tp_counts,
    }


def load_wsdream_dataset2_long(path: Path | str, value_name: str = "value") -> pd.DataFrame:
    """Load a WS-DREAM Dataset #2 file in its four-column long format.

    This full-frame loader is useful for subsets and exploratory work. Use
    ``aggregate_wsdream2_file`` for the complete 40,896,000-row release to
    avoid excessive memory use.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["user_id", "service_id", "time_slice", value_name],
        engine="python",
    )
    df[value_name] = df[value_name].replace(MISSING_SENTINEL, np.nan)

    n_users = df["user_id"].nunique()
    n_services = df["service_id"].nunique()
    n_slices = df["time_slice"].nunique()
    if (n_users, n_services, n_slices) != (
        WSDREAM2_EXPECTED_USERS,
        WSDREAM2_EXPECTED_SERVICES,
        WSDREAM2_EXPECTED_TIMESLICES,
    ):
        import warnings

        warnings.warn(
            f"load_wsdream_dataset2_long: expected "
            f"({WSDREAM2_EXPECTED_USERS} users, {WSDREAM2_EXPECTED_SERVICES} "
            f"services, {WSDREAM2_EXPECTED_TIMESLICES} time slices), found "
            f"({n_users}, {n_services}, {n_slices}). This loader was not "
            f"byte-verified against a real Dataset #2 file -- inspect the "
            f"raw file by hand before trusting this result.",
            stacklevel=2,
        )
    return df
