"""Evaluation protocol (paper §4.2, §4.3): orchestrates the four
conditions across task profiles, candidate pools, and (separately) the
drift sequence.

Practical note this module is built around: the full stable-condition
protocol as specified (6 profiles + ~8 held-out tasks, 30 pools each,
2 of the 4 conditions requiring a real LLM call) is on the order of
several hundred to ~1,000 LLM calls. That is real money and real wall-
clock time, and a network hiccup partway through a long run is a normal
thing to happen, not an edge case. Every run function in this module
therefore writes each completed trial to disk immediately (append-only
CSV) and, on startup, skips any (profile_key, pool_seed, condition)
combination already present in the output file. Killing the process and
re-running the same command resumes rather than restarts.
"""
from __future__ import annotations

import csv
import json
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Any, Iterable

import numpy as np
import pandas as pd

from agentic_selection.agent.controller import AgentController
from agentic_selection.baselines import GLOBAL_FIXED_WEIGHTS, TASK_LOOKUP_TABLE, topsis
from agentic_selection.baselines.lookup_table import get_lookup_weights
from agentic_selection.data.preprocessing import NormalizedCandidatePool, sample_candidate_pool
from agentic_selection.drift.simulate import find_common_top_choice, simulate_drift_sequence
from agentic_selection.evaluation.metrics import adaptation_lag, regret
from agentic_selection.evaluation.storage import CsvStorage
from agentic_selection.tasks import HELD_OUT_TASKS, TASK_PROFILES, HeldOutTask, TaskProfile


@dataclass
class Trial:
    keys: Dict[str, Any]
    execute: Callable[[], Dict[str, Any] | None]


class ExperimentEngine:
    """Decoupled evaluation orchestrator. Isolates thread pooling and storage boundaries
    from the domain-specific trial implementations. Deep module."""
    def __init__(self, storage: CsvStorage, max_workers: int = 1):
        self.storage = storage
        self.max_workers = max_workers
        
    def run(self, trials: Iterable[Trial]) -> pd.DataFrame:
        def _run_trial(trial: Trial):
            if not self.storage.should_run(trial.keys):
                return
            result = trial.execute()
            if result is not None:
                self.storage.record(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for future in concurrent.futures.as_completed([executor.submit(_run_trial, t) for t in trials]):
                future.result()
        return self.storage.load_all()


STABLE_CONDITIONS = ("uniform", "global_every_round", "lookup_every_round", "embedding_knn", "agent_weights_only", "agent_full")
STABLE_RESULT_FIELDS = [
    "run_id", "config_hash", "git_sha", "model", "task_key",
    "task_kind", "task_description", "pool_seed", "condition",
    "regret", "fallback_triggered", "latency_seconds", "api_calls",
    "top_service_id", "strategy", "weights_json", "category",
    "is_synthetic_data", "prompt_tokens", "completion_tokens", "total_duration",
    "weight_l1", "weight_entropy", "max_weight", "top3_overlap",
    "normalized_regret", "ndcg_at_5", "kendall_tau",
]

def run_stable_protocol(
    normalized_df: pd.DataFrame,
    attribute_cols: Sequence[str],
    storage: CsvStorage,
    agent_controller: AgentController,
    run_id: str = "default",
    config_hash: str = "unknown",
    git_sha: str = "unknown",
    model: str = "unknown",
    n_pools: int = 30,
    pool_size: int = 20,
    base_seed: int = 1000,
    is_synthetic_data: bool = False,
    max_workers: int = 1,
    task_profiles: Sequence[TaskProfile] = TASK_PROFILES,
    held_out_tasks: Sequence[HeldOutTask] = HELD_OUT_TASKS,
    conditions: Sequence[str] = STABLE_CONDITIONS,
    global_fixed_weights: Optional[Dict[str, float]] = None,
    lookup_table: Optional[Dict[str, Dict[str, float]]] = None,
    lookup_weights_fn: Optional[Callable[[str], Dict[str, float]]] = None,
) -> pd.DataFrame:
    global_fixed_weights = global_fixed_weights if global_fixed_weights is not None else GLOBAL_FIXED_WEIGHTS
    lookup_table = lookup_table if lookup_table is not None else TASK_LOOKUP_TABLE
    lookup_weights_fn = lookup_weights_fn if lookup_weights_fn is not None else get_lookup_weights

    all_tasks: List[Tuple[str, str, str]] = [
        (p.key, "profile", p.description) for p in task_profiles
    ] + [(f"held_out_{i}", "held_out", t.description) for i, t in enumerate(held_out_tasks)]

    reference_weights_by_key: Dict[str, dict] = {p.key: lookup_table[p.key] for p in task_profiles}
    for i, t in enumerate(held_out_tasks):
        if hasattr(t, "reference_weights"):
            w = {c: getattr(t, "reference_weights").get(c, 0.0) for c in attribute_cols}
            total = sum(w.values())
            if total > 0:
                w = {c: v / total for c, v in w.items()}
            reference_weights_by_key[f"held_out_{i}"] = w
        else:
            reference_weights_by_key[f"held_out_{i}"] = lookup_table[getattr(t, "nearest_profile_key")]

    def _execute_stable(task_key, task_kind, task_description, seed, condition, ref_weights):
        pool_df = sample_candidate_pool(normalized_df, n=pool_size, seed=seed)
        pool = NormalizedCandidatePool(pool_df, attribute_cols)

        if condition in ("global_every_round", "global_fixed"):
            scores = topsis(pool, global_fixed_weights)
            fallback, latency, api_calls = False, 0.0, 0
            top_id = scores.sort_values(ascending=False).index[0]
            strategy = "topsis"
            category_val = None
            w_json = json.dumps(global_fixed_weights)
            prompt_tokens, completion_tokens, total_duration = 0, 0, 0
        elif condition == "uniform":
            w = {c: 1.0 / len(attribute_cols) for c in attribute_cols}
            scores = topsis(pool, w)
            fallback, latency, api_calls = False, 0.0, 0
            top_id = scores.sort_values(ascending=False).index[0]
            strategy = "topsis"
            category_val = None
            w_json = json.dumps(w)
            prompt_tokens, completion_tokens, total_duration = 0, 0, 0
        elif condition in ("lookup_every_round", "lookup_table"):
            w = lookup_weights_fn(task_description if task_kind == "held_out" else task_key, fallback="nearest")
            scores = topsis(pool, w)
            fallback, latency, api_calls = False, 0.0, 0
            top_id = scores.sort_values(ascending=False).index[0]
            strategy = "topsis"
            category_val = None
            w_json = json.dumps(w)
            prompt_tokens, completion_tokens, total_duration = 0, 0, 0
        elif condition == "embedding_knn":
            w = get_lookup_weights(task_description, fallback="embedding_knn")
            w = {c: w.get(c, 0.0) for c in attribute_cols}
            scores = topsis(pool, w)
            fallback, latency, api_calls = False, 0.0, 0
            top_id = scores.sort_values(ascending=False).index[0]
            strategy = "topsis"
            category_val = None
            w_json = json.dumps(w)
            prompt_tokens, completion_tokens, total_duration = 0, 0, 0
        elif condition == "soft_knn":
            w = get_lookup_weights(task_description, fallback="soft_knn")
            w = {c: w.get(c, 0.0) for c in attribute_cols}
            scores = topsis(pool, w)
            fallback, latency, api_calls = False, 0.0, 0
            top_id = scores.sort_values(ascending=False).index[0]
            strategy = "topsis"
            category_val = None
            w_json = json.dumps(w)
            prompt_tokens, completion_tokens, total_duration = 0, 0, 0
        elif condition in ("agent_weights_only", "agent_full"):
            strategy_override = "topsis" if condition == "agent_weights_only" else None
            use_mem = (condition == "agent_full")
            decision = agent_controller.decide(
                task_key, task_description, pool, strategy_override=strategy_override, use_memory=use_mem
            )
            scores = decision.ranking
            fallback = decision.fallback_triggered
            latency = decision.latency_seconds
            api_calls = decision.api_calls
            top_id = decision.top_service_id()
            strategy = decision.strategy
            category_val = decision.category
            w_json = json.dumps(decision.weights)
            prompt_tokens = decision.prompt_tokens
            completion_tokens = decision.completion_tokens
            total_duration = decision.total_duration
        else:
            raise ValueError(f"unknown condition: {condition}")

        r = regret(pool.df, scores, ref_weights, attribute_cols)
        from agentic_selection.evaluation.metrics import (
            weight_l1, weight_entropy, max_weight, top3_overlap, normalized_regret, ndcg_at_k, kendall_tau
        )
        
        if condition in ("global_every_round", "global_fixed", "lookup_every_round", "lookup_table", "uniform", "embedding_knn", "soft_knn"):
            pred_weights = json.loads(w_json)
        else:
            pred_weights = decision.weights
            
        metric_w_l1 = weight_l1(pred_weights, ref_weights, attribute_cols)
        metric_w_ent = weight_entropy(pred_weights, attribute_cols)
        metric_max_w = max_weight(pred_weights, attribute_cols)
        
        ref_scores = pool.df.loc[:, list(attribute_cols)].to_numpy(dtype=float) @ np.array(
            [ref_weights[c] for c in attribute_cols], dtype=float
        )
        ref_ranking = pd.Series(ref_scores, index=pool.df.index)
        
        metric_t3o = top3_overlap(scores, ref_ranking)
        metric_nreg = normalized_regret(pool.df, scores, ref_weights, attribute_cols)
        metric_ndcg = ndcg_at_k(scores, ref_ranking, k=5)
        metric_tau = kendall_tau(scores, ref_ranking)
        
        return {
            "run_id": run_id,
            "config_hash": config_hash,
            "git_sha": git_sha,
            "model": model,
            "task_key": task_key,
            "task_kind": task_kind,
            "task_description": task_description,
            "pool_seed": seed,
            "condition": condition,
            "regret": r,
            "fallback_triggered": fallback,
            "latency_seconds": latency,
            "api_calls": api_calls,
            "top_service_id": top_id,
            "strategy": strategy,
            "weights_json": w_json,
            "category": category_val,
            "is_synthetic_data": is_synthetic_data,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_duration": total_duration,
            "weight_l1": metric_w_l1,
            "weight_entropy": metric_w_ent,
            "max_weight": metric_max_w,
            "top3_overlap": metric_t3o,
            "normalized_regret": metric_nreg,
            "ndcg_at_5": metric_ndcg,
            "kendall_tau": metric_tau,
        }

    trials = []
    for task_key, task_kind, task_description in all_tasks:
        ref_weights = reference_weights_by_key[task_key]
        for pool_i in range(n_pools):
            seed = base_seed + pool_i
            for condition in conditions:
                keys = {"run_id": run_id, "model": model, "task_key": task_key, "pool_seed": seed, "condition": condition}
                trials.append(
                    Trial(
                        keys=keys,
                        execute=lambda tk=task_key, tkind=task_kind, td=task_description, s=seed, c=condition, rw=ref_weights: _execute_stable(tk, tkind, td, s, c, rw)
                    )
                )

    engine = ExperimentEngine(storage, max_workers)
    return engine.run(trials)

DRIFT_CONDITIONS = ("global_fixed", "lookup_table", "lookup_every_round", "global_every_round", "agent_weights_only", "agent_full")
DRIFT_RESULT_FIELDS = [
    "run_id", "config_hash", "git_sha", "model", "task_key",
    "trial_seed", "condition", "target_service_id", "degrade_start_round",
    "n_rounds", "profile", "lag_rounds", "censored", "trace_json",
    "is_synthetic_data", "n_switches", "total_duration",
]

def run_drift_protocol(
    normalized_df: pd.DataFrame,
    attribute_cols: Sequence[str],
    storage: CsvStorage,
    agent_controller: AgentController,
    run_id: str = "default",
    config_hash: str = "unknown",
    git_sha: str = "unknown",
    model: str = "unknown",
    n_trials: int = 5,
    pool_size: int = 20,
    n_rounds: int = 16,
    degrade_start_round: int = 6,
    degradation_profile: str = "sudden",
    degradation_magnitude: float = 0.75,
    static_reevaluation_period: int = 5,
    max_seed_attempts_for_consensus: int = 25,
    base_seed: int = 5000,
    is_synthetic_data: bool = False,
    max_workers: int = 1,
    task_profiles: Sequence[TaskProfile] = TASK_PROFILES,
    degraded_attributes_by_profile: Dict[str, List[str]] | None = None,
    conditions: Sequence[str] = DRIFT_CONDITIONS,
) -> pd.DataFrame:
    def _execute_drift(profile, trial_i, default_degraded, task_key, trial_seed, condition):
        pool = None
        target = None
        for attempt in range(max_seed_attempts_for_consensus):
            candidate_seed = trial_seed * 1000 + attempt
            pool_df = sample_candidate_pool(normalized_df, n=pool_size, seed=candidate_seed)
            candidate_pool = NormalizedCandidatePool(pool_df, attribute_cols)
            methods = {
                "global_fixed": lambda p: topsis(p, GLOBAL_FIXED_WEIGHTS),
                "lookup_table": lambda p: topsis(p, TASK_LOOKUP_TABLE[profile.key]),
            }
            t = find_common_top_choice(candidate_pool, methods)
            if t is not None:
                pool, target = candidate_pool, t
                break
        if pool is None:
            print(
                f"[drift] profile={profile.key} trial={trial_i}: no consensus "
                f"target found in {max_seed_attempts_for_consensus} seed attempts, skipping"
            )
            return None

        seq = simulate_drift_sequence(
            pool.df,
            attribute_cols,
            target_service_id=target,
            degraded_attributes=default_degraded,
            n_rounds=n_rounds,
            degrade_start_round=degrade_start_round,
            profile=degradation_profile,
            magnitude=degradation_magnitude,
            seed=trial_seed,
        )

        if condition == "global_fixed":
            decide_fn = lambda p: topsis(NormalizedCandidatePool(p, attribute_cols), GLOBAL_FIXED_WEIGHTS).sort_values(ascending=False).index[0]
            reeval = static_reevaluation_period
            dur = 0
        elif condition == "global_every_round":
            decide_fn = lambda p: topsis(NormalizedCandidatePool(p, attribute_cols), GLOBAL_FIXED_WEIGHTS).sort_values(ascending=False).index[0]
            reeval = None
            dur = 0
        elif condition == "lookup_table":
            w = TASK_LOOKUP_TABLE[profile.key]
            decide_fn = lambda p, w=w: topsis(NormalizedCandidatePool(p, attribute_cols), w).sort_values(ascending=False).index[0]
            reeval = static_reevaluation_period
            dur = 0
        elif condition == "lookup_every_round":
            w = TASK_LOOKUP_TABLE[profile.key]
            decide_fn = lambda p, w=w: topsis(NormalizedCandidatePool(p, attribute_cols), w).sort_values(ascending=False).index[0]
            reeval = None
            dur = 0
        elif condition in ("agent_weights_only", "agent_full", "rag_agent"):
            strategy_override = "topsis" if condition == "agent_weights_only" else None
            base_ctrl = agent_controller
            fresh_mem_path = Path(base_ctrl.memory.path).parent / f"mem_{run_id}_{condition}_{trial_seed}.jsonl"
            ctrl = AgentController(
                backend=base_ctrl.backend,
                attribute_cols=base_ctrl.attribute_cols,
                memory_path=fresh_mem_path,
                tool_menu=base_ctrl.tool_menu,
                k_memory=base_ctrl.k_memory,
                reasoning_strategy=base_ctrl.reasoning_strategy
            )
            ctrl._llm_cache = base_ctrl._llm_cache
            
            dur = 0
            def _decide_and_record_dur(p):
                nonlocal dur
                dec = ctrl.decide(profile.key, profile.description, NormalizedCandidatePool(p, attribute_cols), strategy_override=strategy_override)
                dur += dec.total_duration
                return dec.top_service_id()
            
            decide_fn = _decide_and_record_dur
            reeval = None
        else:
            raise ValueError(f"unknown condition: {condition}")

        result = adaptation_lag(
            seq.pools, target, degrade_start_round, decide_fn, reevaluation_period=reeval
        )

        trace = result.active_recommendation_trace
        n_switches = sum(1 for i in range(1, len(trace)) if trace[i] != trace[i-1]) if trace else 0
        
        return {
            "run_id": run_id,
            "config_hash": config_hash,
            "git_sha": git_sha,
            "model": model,
            "task_key": task_key,
            "trial_seed": trial_seed,
            "condition": condition,
            "target_service_id": target,
            "degrade_start_round": degrade_start_round,
            "n_rounds": n_rounds,
            "profile": degradation_profile,
            "lag_rounds": result.lag_rounds if result.lag_rounds is not None else "",
            "censored": result.censored,
            "trace_json": str(trace),
            "is_synthetic_data": is_synthetic_data,
            "n_switches": n_switches,
            "total_duration": dur,
        }

    trials = []
    for profile in task_profiles:
        default_degraded = (degraded_attributes_by_profile or {}).get(
            profile.key, profile.dominant_attributes[:2] or profile.dominant_attributes
        )
        task_key = profile.key
        for trial_i in range(n_trials):
            trial_seed = base_seed + trial_i
            for condition in conditions:
                keys = {"run_id": run_id, "model": model, "task_key": task_key, "trial_seed": trial_seed, "condition": condition}
                trials.append(
                    Trial(
                        keys=keys,
                        execute=lambda p=profile, t_i=trial_i, d=default_degraded, tk=task_key, ts=trial_seed, c=condition: _execute_drift(p, t_i, d, tk, ts, c)
                    )
                )

    engine = ExperimentEngine(storage, max_workers)
    return engine.run(trials)
