"""Synthetic fallback data generator.

This module exists for exactly one reason: so the pipeline is always
runnable end-to-end -- for CI, for a quick smoke test, or on a machine
with no internet access -- even if the real QWS download is ever
unreachable. It is NOT a substitute data source for actual results.

Every DataFrame this module produces carries an ``is_synthetic=True``
column, and every entry point in scripts/ that falls back to it prints a
loud, impossible-to-miss warning and refuses to let a report be generated
from it without an explicit ``--allow-synthetic`` flag (see
scripts/06_generate_report.py). Do not submit results computed on this
data as if they were computed on the real QWS dataset.

Per-attribute (min, max, mean, std) below are matched to the real QWS
Dataset v2.0's actual summary statistics, inspected directly from the
verified source file during development (see qws_loader.py for
provenance) -- so marginal ranges/shapes are realistic. Attributes are
sampled *independently* of each other, which real QWS data is not (e.g.
response_time and latency are correlated in the real data); this is a
known, disclosed limitation of the fallback, acceptable because it is
never used as a basis for reported results.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS, SERVICE_ID_COL

# (min, max, mean, std), matched against the real QWS_Dataset_v2.0 file.
_QWS_REFERENCE_STATS: Dict[str, Tuple[float, float, float, float]] = {
    "response_time": (37.0, 4989.67, 383.834, 564.362),
    "availability": (7.0, 100.0, 81.146, 18.701),
    "throughput": (0.10, 43.10, 9.036, 7.731),
    "successability": (8.0, 100.0, 83.887, 19.903),
    "reliability": (33.0, 89.0, 69.783, 8.575),
    "compliance": (33.0, 100.0, 88.434, 10.027),
    "best_practices": (50.0, 95.0, 79.308, 7.817),
    "latency": (0.25, 4140.35, 54.667, 191.709),
    "documentation": (1.0, 97.0, 31.320, 31.509),
}
assert set(_QWS_REFERENCE_STATS) == set(QWS_ATTRIBUTE_COLUMNS)


def _sample_bounded(rng: np.random.Generator, n: int, lo: float, hi: float, mean: float, std: float) -> np.ndarray:
    """Sample n values in [lo, hi] with approximately the given mean/std.

    Uses a Beta distribution (naturally bounded, no clipping needed) fit
    by the method of moments when the target variance is feasible for a
    Beta on this interval; otherwise falls back to a normal distribution
    clipped to [lo, hi] (rare in practice for these particular reference
    stats, but kept as a safety net for robustness).
    """
    rng_span = hi - lo
    m = (mean - lo) / rng_span  # normalized mean in (0, 1)
    v = (std / rng_span) ** 2  # normalized variance
    max_feasible_v = m * (1 - m) * 0.999  # Beta requires v < m(1-m)
    if 0 < m < 1 and 0 < v < max_feasible_v:
        common = m * (1 - m) / v - 1
        alpha = m * common
        beta = (1 - m) * common
        samples = rng.beta(alpha, beta, size=n)
        return lo + samples * rng_span
    # Fallback: clipped normal
    samples = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(samples, lo, hi)


def generate_synthetic_qws(n: int, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic QWS-schema dataset of n rows.

    Returned columns match ``qws_loader.parse_qws_text``'s output for the
    9 numeric attributes plus service_id, with an additional
    ``is_synthetic=True`` column and a placeholder ``service_name`` /
    ``wsdl_address`` so the DataFrame is drop-in compatible with anything
    expecting a real QWS DataFrame's shape.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for col in QWS_ATTRIBUTE_COLUMNS:
        lo, hi, mean, std = _QWS_REFERENCE_STATS[col]
        data[col] = _sample_bounded(rng, n, lo, hi, mean, std)

    df = pd.DataFrame(data)
    df.insert(0, SERVICE_ID_COL, [f"synthetic_{i:04d}" for i in range(n)])
    df["service_name"] = [f"SyntheticService{i}" for i in range(n)]
    df["wsdl_address"] = [f"http://synthetic.invalid/service{i}.wsdl" for i in range(n)]
    df["is_synthetic"] = True
    df = df.set_index(SERVICE_ID_COL, drop=False)
    return df
