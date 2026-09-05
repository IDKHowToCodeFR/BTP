"""Evaluation metrics (paper §4.1).

IMPORTANT DESIGN CORRECTION on adaptation lag vs. the original brief: a
naive protocol that reruns every condition's scoring function fresh on
every round trivially gives every method (static baselines included) an
adaptation lag of ~0, because a stateless scoring function (weighted-sum,
TOPSIS) reacts to new input data the instant it is recomputed -- there is
no "noticing" involved, and this was confirmed empirically while building
this module (see dev notes / README "Corrections vs. the original
brief"). That isn't a meaningful test of anything; it doesn't match what
the paper's own motivation actually claims, which is about deployed
systems not being *re-run* without a person deciding to do so, not about
the arithmetic itself being unable to reflect new numbers.

`adaptation_lag` therefore takes a `reevaluation_period` parameter: static
baselines are re-evaluated only every K rounds (simulating "a human
periodically reruns the MCDM analysis"), while the agent conditions are
re-evaluated every round (simulating "an agent can be re-run automatically
and cheaply on every request", which is the actual mechanism by which an
agentic system could out-adapt a static one -- and which is precisely why
decision cost, H4, is reported alongside this metric rather than treated
as a separate, unrelated number).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd


def regret(
    pool: pd.DataFrame,
    method_scores: pd.Series,
    reference_weights: dict,
    attribute_cols: Sequence[str],
) -> float:
    """Regret = score_under_reference_weights(true best candidate)
    - score_under_reference_weights(method's top-1 choice).

    Both scores are computed as a simple weighted sum under
    `reference_weights` (the researcher-authored "ground truth" priority
    for this task profile, paper §3.7) -- this keeps the regret metric's
    yardstick fixed and independent of whatever scoring formula the
    method under test happens to use internally, which is what makes it
    a fair common currency across weighted-sum, TOPSIS, skyline, and the
    agent (paper §7, limitation: reference weights are researcher-
    authored, not an external objective ground truth -- regret is a
    relative, disclosed-assumption comparison, not an absolute one).

    Returns a non-negative float; 0.0 means the method picked the
    reference-optimal candidate.
    """
    ref_scores = pool.loc[:, list(attribute_cols)].to_numpy(dtype=float) @ np.array(
        [reference_weights[c] for c in attribute_cols], dtype=float
    )
    ref_scores = pd.Series(ref_scores, index=pool.index)

    best_possible = ref_scores.max()
    method_top1 = method_scores.sort_values(ascending=False).index[0]
    chosen_value = ref_scores.loc[method_top1]
    return float(best_possible - chosen_value)


@dataclass
class AdaptationLagResult:
    lag_rounds: Optional[int]  # None = censored (never adapted within the window)
    censored: bool
    active_recommendation_trace: List[object]


def adaptation_lag(
    pools: List[pd.DataFrame],
    target_service_id: object,
    degrade_start_round: int,
    decide_fn: Callable[[pd.DataFrame], object],
    reevaluation_period: Optional[int] = None,
) -> AdaptationLagResult:
    """Rounds after drift onset until the ACTIVE recommendation stops
    being the now-degraded service.

    Parameters
    ----------
    pools : list of per-round pool snapshots (see drift.simulate_drift_sequence)
    target_service_id : the service being artificially degraded
    degrade_start_round : the round index at which degradation begins
    decide_fn : pool snapshot -> top-1 service_id under whatever
        weights/strategy this condition uses (a baseline's weighted-sum/
        TOPSIS call, or an AgentController.decide(...).top_service_id())
    reevaluation_period : None or int
        None: `decide_fn` is called fresh every round (use this for the
        agent conditions -- see module docstring for why this is the
        correct, non-trivial thing to do for an LLM-driven controller
        but would be a meaningless test for a static baseline).
        int K: `decide_fn` is only called at rounds 0, K, 2K, ...; the
        active recommendation is held over between those checkpoints
        (use this for the Global-fixed and Lookup-table baselines).
    """
    n_rounds = len(pools)
    trace: List[object] = []
    active = None
    for i in range(n_rounds):
        should_reevaluate = (
            reevaluation_period is None or i % reevaluation_period == 0 or active is None
        )
        if should_reevaluate:
            active = decide_fn(pools[i])
        trace.append(active)

    for i in range(degrade_start_round, n_rounds):
        if trace[i] != target_service_id:
            return AdaptationLagResult(
                lag_rounds=i - degrade_start_round, censored=False, active_recommendation_trace=trace
            )
    return AdaptationLagResult(lag_rounds=None, censored=True, active_recommendation_trace=trace)


def stability(top_choices: Sequence[object]) -> float:
    """Variance-in-top-choice across repeated identical runs (paper §4.1,
    applicable to the agent only since the LLM's output is not fully
    deterministic even at low temperature). Operationalized as 1 minus
    the fraction of runs that agree with the modal (most common) choice
    -- 0.0 = perfectly stable (always the same top choice), approaching
    1.0 = a different top choice essentially every run.
    """
    if len(top_choices) == 0:
        return 0.0
    counts: dict = {}
    for c in top_choices:
        counts[c] = counts.get(c, 0) + 1
    modal_count = max(counts.values())
    return 1.0 - (modal_count / len(top_choices))


def fallback_trigger_rate(fallback_flags: Sequence[bool]) -> float:
    if len(fallback_flags) == 0:
        return 0.0
    return float(sum(bool(f) for f in fallback_flags) / len(fallback_flags))


@dataclass
class DecisionCostSummary:
    mean_latency_seconds: float
    mean_api_calls: float
    n_decisions: int


def summarize_decision_cost(latencies: Sequence[float], api_calls: Sequence[int]) -> DecisionCostSummary:
    if len(latencies) == 0:
        return DecisionCostSummary(0.0, 0.0, 0)
    return DecisionCostSummary(
        mean_latency_seconds=float(np.mean(latencies)),
        mean_api_calls=float(np.mean(api_calls)),
        n_decisions=len(latencies),
    )


def mean_confidence_interval(values: Sequence[float], confidence: float = 0.95) -> "MeanCI":
    """Normal-approximation 95% CI on the mean. Fine at the sample sizes
    this protocol uses (n=30 pools per condition per profile, paper
    §4.3); for small n or visibly non-normal metrics (e.g. adaptation lag
    with censoring), prefer reporting the bootstrap CI in
    evaluation/report.py instead, or alongside this one.
    """
    from scipy import stats

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return MeanCI(mean=float("nan"), lower=float("nan"), upper=float("nan"), n=0)
    if n == 1:
        return MeanCI(mean=float(arr[0]), lower=float(arr[0]), upper=float(arr[0]), n=1)
    mean = float(np.mean(arr))
    sem = stats.sem(arr)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return MeanCI(mean=mean, lower=mean - margin, upper=mean + margin, n=n)


@dataclass
class MeanCI:
    mean: float
    lower: float
    upper: float
    n: int

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a"
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}] (n={self.n})"
