#!/usr/bin/env python3
"""Download the real QWS dataset (required) and WS-DREAM Dataset #1
(optional/bonus). See data/qws_loader.py and data/wsdream_loader.py for
provenance and format notes.

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --skip-wsdream
    python scripts/01_download_data.py --force   # re-download even if present
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_selection.data.download import ensure_qws, ensure_wsdream_dataset1
from agentic_selection.utils import load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-wsdream", action="store_true", help="Skip the optional WS-DREAM Dataset #1 download")
    parser.add_argument("--force", action="store_true", help="Re-download even if local files already exist")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    data_dir = Path(config["paths"]["data_dir"])

    print(f"== Downloading QWS Dataset v2.0 into {data_dir}/raw/qws ==")
    qws_df, qws_is_synthetic = ensure_qws(data_dir, force=args.force)
    print(f"QWS: {len(qws_df)} rows loaded (synthetic fallback: {qws_is_synthetic})")
    if qws_is_synthetic:
        print(
            "WARNING: QWS download failed and a synthetic fallback was used. "
            "The pipeline will run, but do not use results computed on this "
            "data for anything you intend to report. Check your network / "
            "the source URL and re-run this script.",
            file=sys.stderr,
        )

    if not args.skip_wsdream:
        print(f"\n== Downloading WS-DREAM Dataset #1 (secondary/bonus) into {data_dir}/raw/wsdream1 ==")
        wsdream1, available = ensure_wsdream_dataset1(data_dir, force=args.force)
        print(f"WS-DREAM Dataset #1 available: {available}")

    print("\nDone. Next: python scripts/02_prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
