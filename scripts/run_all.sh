#!/usr/bin/env bash
# Runs the full pipeline in order. Stops on the first failure.
# Pass extra args to forward to the experiment scripts, e.g.:
#   ./scripts/run_all.sh --yes --n-pools 3   (cheap smoke test with confirmation skipped)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/6: download data ==="
python scripts/01_download_data.py

echo "=== 2/6: prepare data ==="
python scripts/02_prepare_data.py

echo "=== 3/6: validate ==="
python scripts/03_validate_baselines.py

echo "=== 4/6: stable experiment ==="
python scripts/04_run_stable_experiment.py "$@"

echo "=== 5/6: drift experiment ==="
python scripts/05_run_drift_experiment.py "$@"

echo "=== 6/6: report ==="
python scripts/06_generate_report.py

echo "=== Done ==="
