"""Orchestrates fetching real datasets, with an explicit, loud fallback to
synthetic data if (and only if) the real download genuinely fails. Never
falls back silently: every fallback path prints a clearly-flagged warning
and the returned/written data is marked ``is_synthetic=True``.

This is invoked by scripts/01_download_data.py; see that script for the
CLI entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

from agentic_selection.data.qws_loader import download_qws, load_qws, QWS_PRIMARY_URL
from agentic_selection.data.synthetic import generate_synthetic_qws
from agentic_selection.data.wsdream_loader import download_wsdream_dataset1, load_wsdream_dataset1

_BANNER = "=" * 78


def _warn_synthetic_fallback(dataset_name: str, error: Exception) -> None:
    print(_BANNER, file=sys.stderr)
    print(f"WARNING: could not download real {dataset_name} data.", file=sys.stderr)
    print(f"  Reason: {error!r}", file=sys.stderr)
    print(
        "  Falling back to a SYNTHETIC dataset with matched marginal "
        "statistics. This is fine for a smoke test / CI run, but results "
        "computed on synthetic data must never be reported as if they "
        "came from the real dataset. Check your network connection or "
        "the source URL and re-run this script to get real data.",
        file=sys.stderr,
    )
    print(_BANNER, file=sys.stderr)


def ensure_qws(data_dir: Path | str, n_synthetic_fallback: int = 2507, force: bool = False):
    """Ensure a local QWS dataset is available under data_dir/raw/qws/.

    Returns (DataFrame, is_synthetic: bool).
    """
    data_dir = Path(data_dir)
    raw_path = data_dir / "raw" / "qws" / "QWS_Dataset_v2.txt"

    if force or not raw_path.exists():
        try:
            print(f"Downloading QWS dataset from {QWS_PRIMARY_URL} ...")
            download_qws(raw_path)
            print(f"  saved to {raw_path}")
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a network call
            _warn_synthetic_fallback("QWS", e)
            df = generate_synthetic_qws(n=n_synthetic_fallback, seed=0)
            synth_path = data_dir / "processed" / "qws_synthetic_fallback.csv"
            synth_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(synth_path, index=False)
            return df, True

    df = load_qws(raw_path)
    df["is_synthetic"] = False
    return df, False


def ensure_wsdream_dataset1(data_dir: Path | str, force: bool = False):
    """Ensure a local copy of WS-DREAM Dataset #1 is available. Unlike
    QWS, this dataset is secondary/bonus in this project's design (see
    wsdream_loader.py docstring), so on failure we skip rather than
    fabricate a synthetic replacement -- a missing secondary dataset
    should just mean "skip the bonus analysis", not "silently swap in
    fake data for it".

    Returns (dict-of-DataFrames or None, available: bool).
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw" / "wsdream1"

    have_all = raw_dir.exists() and all((raw_dir / f).exists() for f in
        ("userlist.txt", "wslist.txt", "rtMatrix.txt", "tpMatrix.txt"))

    if force or not have_all:
        try:
            print("Downloading WS-DREAM Dataset #1 (339 users x 5825 services) ...")
            download_wsdream_dataset1(raw_dir)
            print(f"  saved to {raw_dir}")
        except Exception as e:  # noqa: BLE001
            print(_BANNER, file=sys.stderr)
            print(f"WARNING: could not download WS-DREAM Dataset #1: {e!r}", file=sys.stderr)
            print(
                "  This is a secondary/bonus dataset in this project -- "
                "core experiments (QWS-based) are unaffected. Skipping.",
                file=sys.stderr,
            )
            print(_BANNER, file=sys.stderr)
            return None, False

    return load_wsdream_dataset1(raw_dir), True
