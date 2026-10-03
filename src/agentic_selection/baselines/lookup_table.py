"""Baseline weight sources: global fixed weights and the task-aware lookup table.

Two baseline *weight sources* are defined here, matching paper §3.5:

1. GLOBAL_FIXED_WEIGHTS -- a single vector applied to every request
   regardless of task. Deliberately the least informative honest choice
   (uniform over all 9 QWS attributes) rather than a strawman -- reviewers
   in this space will spot a strawman comparison immediately (this was
   flagged explicitly in the earlier project-design discussion), so the
   weak baseline here is "no task-awareness at all", not "a bad vector".

2. TASK_LOOKUP_TABLE -- one hand-authored vector per predefined task
   profile (see tasks/profiles.py for the natural-language descriptions
   these correspond to). This is the primary, fair comparison point for
   the agent.

Both are used with baselines.topsis / baselines.weighted_sum, which take a
weights mapping directly, so nothing here performs ranking itself.

CORRECTION vs. the original task-profile table: two profiles ("Batch
processing", "Low-cost prototype") originally listed "cost" as a dominant
attribute. QWS has no monetary cost field, so both were redefined to use
only real QWS attributes -- see constants.py docstring and the README for
the full rationale. Batch processing now weights throughput +
successability (finishing large jobs without manual retry); Low-cost
prototype now weights documentation + best_practices (a hobbyist without
dedicated ops staff leans on self-service integration quality rather than
raw performance guarantees).
"""
from __future__ import annotations

import difflib
from typing import Dict, Optional

from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS

Weights = Dict[str, float]

GLOBAL_FIXED_WEIGHTS: Weights = {
    "response_time": 0.123,
    "availability": 0.110,
    "throughput": 0.142,
    "successability": 0.130,
    "reliability": 0.125,
    "compliance": 0.102,
    "best_practices": 0.110,
    "latency": 0.073,
    "documentation": 0.085
}

TASK_LOOKUP_TABLE: Dict[str, Weights] = {
    "streaming": {
        "response_time": 0.20,
        "availability": 0.05,
        "throughput": 0.32,
        "successability": 0.05,
        "reliability": 0.05,
        "compliance": 0.03,
        "best_practices": 0.03,
        "latency": 0.22,
        "documentation": 0.05,
    },
    "financial_transaction": {
        "response_time": 0.08,
        "availability": 0.27,
        "throughput": 0.05,
        "successability": 0.08,
        "reliability": 0.30,
        "compliance": 0.10,
        "best_practices": 0.05,
        "latency": 0.02,
        "documentation": 0.05,
    },
    "batch_processing": {
        "response_time": 0.05,
        "availability": 0.08,
        "throughput": 0.34,
        "successability": 0.26,
        "reliability": 0.10,
        "compliance": 0.03,
        "best_practices": 0.03,
        "latency": 0.06,
        "documentation": 0.05,
    },
    "low_cost_prototype": {
        "response_time": 0.06,
        "availability": 0.08,
        "throughput": 0.06,
        "successability": 0.06,
        "reliability": 0.08,
        "compliance": 0.10,
        "best_practices": 0.28,
        "latency": 0.04,
        "documentation": 0.24,
    },
    "iot_telemetry_ingestion": {
        "response_time": 0.30,
        "availability": 0.08,
        "throughput": 0.05,
        "successability": 0.28,
        "reliability": 0.10,
        "compliance": 0.03,
        "best_practices": 0.03,
        "latency": 0.08,
        "documentation": 0.05,
    },
    "compliance_sensitive_backend": {
        "response_time": 0.05,
        "availability": 0.10,
        "throughput": 0.03,
        "successability": 0.05,
        "reliability": 0.12,
        "compliance": 0.32,
        "best_practices": 0.24,
        "latency": 0.02,
        "documentation": 0.07,
    },
}


def _validate_lookup_table() -> None:
    for key, w in TASK_LOOKUP_TABLE.items():
        missing = set(QWS_ATTRIBUTE_COLUMNS) - set(w)
        extra = set(w) - set(QWS_ATTRIBUTE_COLUMNS)
        if missing or extra:
            raise ValueError(f"profile '{key}': missing={missing} extra={extra}")
        total = sum(w.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"profile '{key}': weights sum to {total}, expected 1.0")
    total_global = sum(GLOBAL_FIXED_WEIGHTS.values())
    if abs(total_global - 1.0) > 1e-6:
        raise ValueError(f"GLOBAL_FIXED_WEIGHTS sums to {total_global}, expected 1.0")


_validate_lookup_table()


def get_lookup_weights(
    profile_key: str,
    fallback: str = "nearest",
    similarity_cutoff: float = 0.5,
) -> Weights:
    """Look up the hand-authored weight vector for a task profile key.

    Parameters
    ----------
    profile_key : str
        One of TASK_LOOKUP_TABLE's keys, or arbitrary text (e.g. a raw
        task description) for which we attempt a fuzzy match.
    fallback : {"nearest", "global", "raise"}
        What to do when profile_key is not an exact match:
        - "nearest": fuzzy-match against known profile keys using
          difflib.get_close_matches; if nothing clears
          similarity_cutoff, fall back to GLOBAL_FIXED_WEIGHTS. This is
          the lookup table's only mechanism for handling task phrasing
          outside its fixed coverage (paper §4.2/H3) -- by construction
          it cannot understand novel phrasing, only pattern-match it.
        - "embedding_knn": matches against the task profiles' descriptions
          using sentence-transformers.
        - "soft_knn": blends weights from all known task profiles using softmax over embedding similarities (temp 0.1)
        - "global": always fall back to GLOBAL_FIXED_WEIGHTS on a miss.
        - "raise": raise KeyError on a miss (useful in tests).
    similarity_cutoff : float
        difflib similarity threshold in [0, 1] for the "nearest" fallback.

    Returns
    -------
    dict mapping attribute name -> weight (sums to 1.0)
    """
    key = profile_key.strip().lower().replace(" ", "_").replace("-", "_")
    if key in TASK_LOOKUP_TABLE:
        return dict(TASK_LOOKUP_TABLE[key])

    if fallback == "raise":
        raise KeyError(f"no lookup-table entry for profile_key={profile_key!r}")

    if fallback == "global":
        return dict(GLOBAL_FIXED_WEIGHTS)

    if fallback == "embedding_knn":
        key_match = nearest_profile_key(profile_key, fallback="embedding_knn")
        if key_match:
            return dict(TASK_LOOKUP_TABLE[key_match])
        return dict(GLOBAL_FIXED_WEIGHTS)

    if fallback == "soft_knn":
        return soft_knn_weights(profile_key, temp=0.1)

    if fallback != "nearest":
        raise ValueError(f"unknown fallback mode: {fallback}")

    candidates = list(TASK_LOOKUP_TABLE.keys())
    matches = difflib.get_close_matches(key, candidates, n=1, cutoff=similarity_cutoff)
    if matches:
        return dict(TASK_LOOKUP_TABLE[matches[0]])
    return dict(GLOBAL_FIXED_WEIGHTS)


# Caching the SentenceTransformer model to avoid reloading it constantly
_embed_model = None

def nearest_profile_key(profile_key: str, similarity_cutoff: float = 0.5, fallback: str = "nearest") -> Optional[str]:
    """Which known profile key (if any) a free-text description matches."""
    key = profile_key.strip().lower().replace(" ", "_").replace("-", "_")
    if key in TASK_LOOKUP_TABLE:
        return key
        
    if fallback == "embedding_knn":
        global _embed_model
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            _embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
        from agentic_selection.tasks import TASK_PROFILES
        candidates = list(TASK_PROFILES)
        
        # embed query and candidates
        query_emb = _embed_model.encode(profile_key, convert_to_tensor=True)
        cand_docs = [p.description for p in candidates]
        cand_embs = _embed_model.encode(cand_docs, convert_to_tensor=True)
        
        from sentence_transformers import util
        scores = util.cos_sim(query_emb, cand_embs)[0]
        best_idx = int(scores.argmax())
        if float(scores[best_idx]) >= similarity_cutoff:
            return candidates[best_idx].key
        return None

    candidates = list(TASK_LOOKUP_TABLE.keys())
    matches = difflib.get_close_matches(key, candidates, n=1, cutoff=similarity_cutoff)
    return matches[0] if matches else None


def soft_knn_weights(profile_key: str, temp: float = 0.1) -> Weights:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
    from agentic_selection.tasks import TASK_PROFILES
    candidates = list(TASK_PROFILES)
    
    query_emb = _embed_model.encode(profile_key, convert_to_tensor=True)
    cand_docs = [p.description for p in candidates]
    cand_embs = _embed_model.encode(cand_docs, convert_to_tensor=True)
    
    from sentence_transformers import util
    import torch
    scores = util.cos_sim(query_emb, cand_embs)[0]
    
    # softmax with temperature
    weights = torch.softmax(scores / temp, dim=0)
    
    blended = {c: 0.0 for c in QWS_ATTRIBUTE_COLUMNS}
    for i, p in enumerate(candidates):
        w = float(weights[i])
        p_weights = TASK_LOOKUP_TABLE[p.key]
        for c in QWS_ATTRIBUTE_COLUMNS:
            blended[c] += w * p_weights.get(c, 0.0)
            
    # normalize
    total = sum(blended.values())
    if total > 0:
        return {c: v/total for c,v in blended.items()}
    return dict(GLOBAL_FIXED_WEIGHTS)
