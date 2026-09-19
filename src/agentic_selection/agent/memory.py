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
        self._records: Optional[List[MemoryRecord]] = None

    def append(self, record: MemoryRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(record.to_json_line() + "\n")
        if self._records is not None:
            self._records.append(record)

    def load_all(self) -> List[MemoryRecord]:
        if self._records is not None:
            return self._records
        
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(MemoryRecord.from_dict(json.loads(line)))
        self._records = records
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

    def nearest(self, query_vector: np.ndarray, k: int = 3, query_task: str = "") -> List[MemoryRecord]:
        from difflib import SequenceMatcher
        records = self.load_all()
        if not records:
            return []
        
        # Pass 1: Fast euclidean distance over the perception vector for all records
        fast_scored = []
        for r in records:
            v = np.array(r.perception_vector, dtype=float)
            if v.shape != query_vector.shape:
                continue
            dist = float(np.linalg.norm(v - query_vector))
            fast_scored.append((dist, r))
        
        # Keep only the top 50 closest records to avoid O(N) slow text comparisons
        fast_scored.sort(key=lambda t: t[0])
        candidates = fast_scored[:50]

        # Pass 2: Heavy text similarity only on the top candidates
        final_scored = []
        for dist, r in candidates:
            text_sim = 0.0
            if query_task:
                text_sim = SequenceMatcher(None, query_task, r.task_description).ratio()
            # We want to minimize score. Max text_sim is 1.0. Subtract weighted text similarity.
            score = dist - (text_sim * 10.0)
            final_scored.append((score, r))
            
        final_scored.sort(key=lambda t: t[0])
        return [r for _, r in final_scored[:k]]


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


def format_digest(records: List[MemoryRecord], max_chars: int = 2000) -> str:
    """Condense a handful of past records into a JSON digest for
    the reasoning prompt as few-shot examples."""
    if not records:
        return ""
    lines = []
    for r in records:
        example = {
            "weights": r.weights,
            "strategy": r.strategy,
            "justification": r.justification
        }
        lines.append(f'Task: "{r.task_description}"\nOutput:\n```json\n{json.dumps(example, indent=2)}\n```')
    digest = "\n\n".join(lines)
    return digest[:max_chars]
