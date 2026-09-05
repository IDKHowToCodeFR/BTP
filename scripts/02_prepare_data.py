#!/usr/bin/env python3
"""Normalize the raw downloaded QWS dataset (min-max + cost-inversion,
paper §3.3) and cache the result to data/processed/. Also builds the
WS-DREAM Dataset #1 per-service candidate pool if that data is present.

Usage:
    python scripts/02_prepare_data.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.data.preprocessing import normalize_benefit_oriented
from agentic_selection.data.qws_loader import load_qws
from agentic_selection.data.wsdream_loader import load_wsdream_dataset1, wsdream1_to_candidate_pool
from agentic_selection.utils import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    data_dir = Path(config["paths"]["data_dir"])

    qws_raw_path = data_dir / "raw" / "qws" / "QWS_Dataset_v2.txt"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if qws_raw_path.exists():
        print(f"Loading QWS from {qws_raw_path} ...")
        df = load_qws(qws_raw_path)
        df["is_synthetic"] = False
        print(f"  {len(df)} rows")
        norm = normalize_benefit_oriented(df)
        out_path = processed_dir / "qws_normalized.csv"
        norm.to_csv(out_path)
        print(f"  wrote normalized data to {out_path}")
    else:
        synth_path = processed_dir / "qws_synthetic_fallback.csv"
        if synth_path.exists():
            print(
                f"No real QWS raw file found, but a synthetic fallback exists "
                f"at {synth_path} (from a previous run of "
                f"scripts/01_download_data.py). Normalizing that instead -- "
                f"remember this is NOT real data.",
                file=sys.stderr,
            )
            import pandas as pd

            df = pd.read_csv(synth_path).set_index("service_id", drop=False)
            norm = normalize_benefit_oriented(df)
            norm.to_csv(processed_dir / "qws_normalized.csv")
        else:
            print(
                "No QWS data found at all. Run scripts/01_download_data.py first.",
                file=sys.stderr,
            )
            return 1

    wsdream1_dir = data_dir / "raw" / "wsdream1"
    if all((wsdream1_dir / f).exists() for f in ("userlist.txt", "wslist.txt", "rtMatrix.txt", "tpMatrix.txt")):
        print(f"\nLoading WS-DREAM Dataset #1 from {wsdream1_dir} ...")
        wsdream1 = load_wsdream_dataset1(wsdream1_dir)
        pool = wsdream1_to_candidate_pool(wsdream1, min_observations=30)
        out_path = processed_dir / "wsdream1_candidate_pool.csv"
        pool.to_csv(out_path)
        print(f"  {len(pool)} services with >=30 observations, wrote to {out_path}")
    else:
        print("\nWS-DREAM Dataset #1 not found locally -- skipping (it's optional/bonus).")

    print("\nDone. Next: python scripts/03_validate_baselines.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
