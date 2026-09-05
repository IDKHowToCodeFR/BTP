"""Memory module (paper §3.6): a persistent, append-mostly log of past
(task description, perception summary, decision, outcome) tuples, with
simple nearest-neighbor retrieval over the perception vector. No vector
database is used, matching the paper's stated design ("no vector
database is used given the small scale of this study") -- this is a
JSON Lines file plus an O(n) linear scan, which is more than fast enough
at the scale this project operates at (tens of thousands of decisions at
most).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MemoryRecord:
    record_id: str
    timestamp: str
    task_description: str
    perception_vector: List[float]
    weights: Dict[str, float]
    strategy: str
    justification: str
    fallback_triggered: bool
    fallback_reason: Optional[str]
    outcome: Optional[Dict[str, Any]] = None  # filled in later via record_outcome()

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_dict(d: dict) -> "MemoryRecord":
        return MemoryRecord(**d)


def new_record_id() -> str:
    return uuid.uuid4().hex[:12]


class MemoryStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, record: MemoryRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(record.to_json_line() + "\n")

    def load_all(self) -> List[MemoryRecord]:
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(MemoryRecord.from_dict(json.loads(line)))
        return records

    def record_outcome(self, record_id: str, outcome: Dict[str, Any]) -> bool:
        """Attach an observed outcome to a previously-logged record. Since
        this is a JSONL append log, "updating" means rewriting the whole
        file with the matching line replaced -- documented as an accepted
        trade-off given this project's scale (paper §3.6). Returns True
        if a matching record was found and updated.
        """
        records = self.load_all()
        found = False
        for r in records:
            if r.record_id == record_id:
                r.outcome = outcome
                found = True
        if found:
            with open(self.path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(r.to_json_line() + "\n")
        return found

    def nearest(self, query_vector: np.ndarray, k: int = 3) -> List[MemoryRecord]:
        records = self.load_all()
        if not records:
            return []
        scored = []
        for r in records:
            v = np.array(r.perception_vector, dtype=float)
            if v.shape != query_vector.shape:
                # Different attribute schema than the current query (e.g.
                # a different dataset was used for that past decision) --
                # not comparable, skip rather than error.
                continue
            dist = float(np.linalg.norm(v - query_vector))
            scored.append((dist, r))
        scored.sort(key=lambda t: t[0])
        return [r for _, r in scored[:k]]


def make_record(
    task_description: str,
    perception_vector: np.ndarray,
    weights: Dict[str, float],
    strategy: str,
    justification: str,
    fallback_triggered: bool,
    fallback_reason: Optional[str],
) -> MemoryRecord:
    return MemoryRecord(
        record_id=new_record_id(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        task_description=task_description,
        perception_vector=[float(x) for x in perception_vector],
        weights=weights,
        strategy=strategy,
        justification=justification,
        fallback_triggered=fallback_triggered,
        fallback_reason=fallback_reason,
    )


def format_digest(records: List[MemoryRecord], max_chars: int = 800) -> str:
    """Condense a handful of past records into a short text digest for
    the reasoning prompt (paper §3.6, component (d))."""
    if not records:
        return ""
    lines = []
    for r in records:
        outcome_str = ""
        if r.outcome:
            outcome_str = f" | observed outcome: {json.dumps(r.outcome)}"
        top_weights = sorted(r.weights.items(), key=lambda kv: -kv[1])[:3]
        w_str = ", ".join(f"{k}={v:.2f}" for k, v in top_weights)
        lines.append(
            f'- Task: "{r.task_description[:120]}" -> strategy={r.strategy}, '
            f"top weights: {w_str}{outcome_str}"
        )
    digest = "\n".join(lines)
    return digest[:max_chars]
