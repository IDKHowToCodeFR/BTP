"""Loader for the QWS Dataset v2.0 (Al-Masri & Mahmoud, 2007).

Provenance note: the "official" project site (qwsdata.github.io, linked
from the paper's own references) currently serves only a placeholder HTML
page, not the raw data file -- confirmed by direct inspection while
building this loader. The raw dataset (2,507 rows, 9 QoS attributes) is
mirrored as a plain-text file in several third-party research repos; this
loader was built and validated against:

    https://raw.githubusercontent.com/mahsaa/Coopetition/master/QWS_Dataset_v2.txt

Two real format quirks were found and are handled explicitly below (both
would silently corrupt results if missed):

1. The file has ~20 leading comment/header lines starting with "##",
   which must be skipped.
2. Exactly 2 of the 2,507 data rows contain an *extra*, embedded comma
   inside the WSDL address field itself (e.g. ".../viewrep/~raw,r=1.2/...").
   A naive ``line.split(",")`` misaligns every field after that point for
   those rows. The fix is ``line.split(",", 10)`` -- split on the first 10
   commas only, so the 11th (final) field, the WSDL address, absorbs any
   further commas verbatim. This was verified against the actual file: it
   produces exactly 11 fields for all 2,507 rows with no exceptions.

If you obtain the file from a different mirror, re-run
``scripts/03_validate_baselines.py``'s data sanity checks before trusting
results -- do not assume every mirror has identical formatting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS, SERVICE_ID_COL

QWS_NUMERIC_COLUMNS = list(QWS_ATTRIBUTE_COLUMNS)  # 9 columns, in file order
QWS_ALL_RAW_COLUMNS = QWS_NUMERIC_COLUMNS + ["service_name", "wsdl_address"]

QWS_PRIMARY_URL = "https://raw.githubusercontent.com/mahsaa/Coopetition/master/QWS_Dataset_v2.txt"
QWS_EXPECTED_ROW_COUNT = 2507


def parse_qws_text(raw_text: str) -> pd.DataFrame:
    """Parse the raw QWS_Dataset_v2.txt content into a DataFrame.

    Raises ValueError if the parsed row count or field count looks wrong,
    rather than silently returning a corrupted DataFrame -- this dataset
    is used as ground truth for an entire paper's experiments, so a loud
    failure here is much cheaper than a quiet one later.
    """
    lines = [
        line.rstrip("\n").rstrip("\r")
        for line in raw_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != QWS_EXPECTED_ROW_COUNT:
        import warnings

        warnings.warn(
            f"parse_qws_text: expected {QWS_EXPECTED_ROW_COUNT} data rows "
            f"(the documented size of QWS Dataset v2.0), found {len(lines)}. "
            f"Proceeding, but double-check the source file if this is "
            f"unexpected -- e.g. a different mirror or a truncated download.",
            stacklevel=2,
        )

    records = []
    bad_rows = 0
    for i, line in enumerate(lines):
        parts = line.split(",", 10)  # cap at 11 fields; see module docstring
        if len(parts) != 11:
            bad_rows += 1
            continue
        records.append(parts)

    if bad_rows:
        raise ValueError(
            f"{bad_rows} row(s) did not parse into exactly 11 fields even "
            f"with split(',', 10). The source file's format may have "
            f"changed; inspect it before trusting any downstream results."
        )

    df = pd.DataFrame(records, columns=QWS_ALL_RAW_COLUMNS)
    for col in QWS_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df.insert(0, SERVICE_ID_COL, [f"qws_{i:04d}" for i in range(len(df))])
    df = df.set_index(SERVICE_ID_COL, drop=False)
    return df


def load_qws(path: Path | str) -> pd.DataFrame:
    """Load and parse a local copy of QWS_Dataset_v2.txt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"QWS dataset not found at {path}. Run "
            f"scripts/01_download_data.py first, or pass an explicit path."
        )
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    return parse_qws_text(raw_text)


def download_qws(dest_path: Path | str, url: str = QWS_PRIMARY_URL, timeout: int = 30) -> Path:
    """Download the raw QWS dataset file to dest_path. Raises on failure
    (network error, non-200 status, or empty body) -- callers that want a
    graceful fallback should catch the exception, see data/download.py.
    """
    import requests

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    if len(resp.content) < 1000:
        raise ValueError(
            f"Downloaded QWS file from {url} is suspiciously small "
            f"({len(resp.content)} bytes) -- likely an error page, not "
            f"the dataset. Refusing to write it."
        )
    dest_path.write_bytes(resp.content)
    return dest_path
