"""Central place to document this project's seeding convention.

Base seed ranges are deliberately kept far apart across different parts
of the pipeline so that, e.g., a stable-protocol pool sample and a
drift-protocol pool sample can never accidentally collide on the same
seed and be mistaken for the same draw:

    Stable protocol pool sampling : base_seed=1000, +1 per pool index
    Drift protocol trial sampling : base_seed=5000, +1 per trial, then
                                     *1000 + attempt for consensus search
    Synthetic data fallback        : seed=0 (fixed, deterministic; only
                                      one synthetic dataset is ever needed)

These are just the *defaults* used by evaluation/protocol.py and
data/synthetic.py -- pass explicit seeds yourself for anything you want
independently reproducible outside those defaults.
"""
from __future__ import annotations

STABLE_PROTOCOL_BASE_SEED = 1000
DRIFT_PROTOCOL_BASE_SEED = 5000
SYNTHETIC_DATA_SEED = 0
