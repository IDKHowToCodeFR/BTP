"""Skyline (Pareto-optimal) filtering.

As with weighted_sum and topsis, this assumes attribute_cols are already
normalized to [0, 1] and benefit-oriented (higher = better) -- see
data/preprocessing.py. Candidate i dominates candidate j iff i is >= j on
every attribute in attribute_cols and strictly > j on at least one. The
skyline is the set of candidates dominated by nobody.

Memory note: a fully vectorized pairwise dominance check needs an
(n, n, m) boolean tensor, which is fine for the paper's protocol (pools
of n=20) but would use several hundred MB if run over the full QWS
dataset (n=2507) without chunking. ``skyline()`` therefore chunks over
the first axis; this changes nothing about the result, only peak memory.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from agentic_selection.baselines.topsis import topsis


def skyline(
    df: pd.DataFrame,
    attribute_cols: Sequence[str],
    chunk_size: int = 500,
) -> pd.Index:
    """Return the index labels of the non-dominated (skyline) rows of ``df``.

    Parameters
    ----------
    df : DataFrame
    attribute_cols : sequence of str
        Columns to compute dominance over. Assumed benefit-oriented [0, 1].
    chunk_size : int
        Rows processed per batch when computing pairwise dominance, to
        bound peak memory on large candidate pools. Does not affect the
        result, only performance/memory.

    Returns
    -------
    pd.Index
        Subset of df.index that is Pareto-optimal (non-dominated).
    """
    if len(attribute_cols) == 0:
        raise ValueError("attribute_cols must be non-empty")
    for c in attribute_cols:
        if c not in df.columns:
            raise KeyError(f"attribute column '{c}' not found in df.columns={list(df.columns)}")
    n = len(df)
    if n == 0:
        return df.index[:0]
    if n == 1:
        return df.index

    X = df.loc[:, list(attribute_cols)].to_numpy(dtype=float)
    if np.isnan(X).any():
        bad_cols = df.loc[:, list(attribute_cols)].columns[
            df.loc[:, list(attribute_cols)].isnull().any()
        ].tolist()
        raise ValueError(
            f"skyline received NaN values in columns {bad_cols}; "
            f"impute or drop before filtering (see data/preprocessing.py)."
        )

    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end]  # (c, m)
        # ge[k, j] = True if chunk[k] >= X[j] on every attribute
        ge = np.all(chunk[:, None, :] >= X[None, :, :], axis=2)  # (c, n)
        gt = np.any(chunk[:, None, :] > X[None, :, :], axis=2)  # (c, n)
        dominates = ge & gt  # (c, n): does chunk[k] dominate X[j]?
        # zero out self-comparison (row start+k vs itself)
        for k in range(end - start):
            dominates[k, start + k] = False
        dominated_by_chunk = dominates.any(axis=0)  # (n,) any candidate dominated by someone in this chunk
        dominated |= dominated_by_chunk

    return df.index[~dominated]


def skyline_then_topsis(
    df: pd.DataFrame,
    weights: Mapping[str, float],
    attribute_cols: Sequence[str],
    chunk_size: int = 500,
) -> pd.Series:
    """Filter to the skyline set, then rank that subset with TOPSIS.

    Returns a full-length Series aligned to ``df.index`` so this function
    has the same contract as ``weighted_sum`` and ``topsis`` (higher =
    better, safe to argmax/sort directly), which matters because the
    agent's action module and the evaluation harness treat all three
    strategies uniformly. Non-skyline candidates are assigned a sentinel
    score of -1.0, which is guaranteed to be lower than any valid TOPSIS
    closeness coefficient (always in [0, 1]), so a plain ``argmax`` /
    ``sort_values(ascending=False)`` over the returned Series always
    prefers a skyline member when one exists.
    """
    sky_idx = skyline(df, attribute_cols, chunk_size=chunk_size)
    sky_scores = topsis(df.loc[sky_idx], weights, attribute_cols)

    full = pd.Series(-1.0, index=df.index, name="skyline_topsis_score")
    full.loc[sky_idx] = sky_scores
    return full


def rank_skyline_then_topsis(
    df: pd.DataFrame,
    weights: Mapping[str, float],
    attribute_cols: Sequence[str],
) -> pd.DataFrame:
    scores = skyline_then_topsis(df, weights, attribute_cols)
    out = df.copy()
    out["score"] = scores
    return out.sort_values("score", ascending=False)
