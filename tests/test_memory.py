from __future__ import annotations

import numpy as np
import pytest

from agentic_selection.agent.memory import MemoryStore, make_record, format_digest


def test_memory_append_and_load_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    r = make_record("task A", np.array([1.0, 2.0]), {"a": 1.0}, "topsis", "why", False, None)
    store.append(r)
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].task_description == "task A"
    assert loaded[0].weights == {"a": 1.0}


def test_memory_nearest_returns_closest_by_euclidean_distance(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    store.append(make_record("near", np.array([1.0, 1.0]), {}, "topsis", "", False, None))
    store.append(make_record("far", np.array([10.0, 10.0]), {}, "topsis", "", False, None))
    results = store.nearest(np.array([1.1, 1.1]), k=1)
    assert len(results) == 1
    assert results[0].task_description == "near"


def test_memory_nearest_skips_mismatched_vector_shapes(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    store.append(make_record("2d", np.array([1.0, 1.0]), {}, "topsis", "", False, None))
    results = store.nearest(np.array([1.0, 1.0, 1.0]), k=3)  # 3-d query, no comparable records
    assert results == []


def test_memory_nearest_on_empty_store_returns_empty(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    assert store.nearest(np.array([1.0]), k=3) == []


def test_memory_record_outcome_updates_existing_record(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    r = make_record("task", np.array([1.0]), {}, "topsis", "", False, None)
    store.append(r)
    found = store.record_outcome(r.record_id, {"regret": 0.05})
    assert found
    loaded = store.load_all()
    assert loaded[0].outcome == {"regret": 0.05}


def test_memory_record_outcome_missing_id_returns_false(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    assert store.record_outcome("nonexistent_id", {"regret": 0.0}) is False


def test_format_digest_empty_records_returns_empty_string():
    assert format_digest([]) == ""


def test_format_digest_includes_task_and_outcome(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    r = make_record("streaming task", np.array([1.0]), {"throughput": 0.5}, "topsis", "j", False, None)
    store.append(r)
    r2 = store.load_all()[0]
    r2.outcome = {"regret": 0.1}
    digest = format_digest([r2])
    assert "streaming task" in digest
    assert "throughput" in digest
