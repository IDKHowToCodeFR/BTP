"""Experiment storage abstraction (architecture deepening).

Provides the `ExperimentStorage` interface and a `CsvStorage` adapter
for resumable tracking of experiment trials.
"""
from __future__ import annotations

import abc
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


class ExperimentStorage(abc.ABC):
    """Deep interface for tracking experiment results and resumability."""

    @abc.abstractmethod
    def should_run(self, key_dict: Dict[str, Any]) -> bool:
        """Return True if the trial defined by key_dict has NOT been recorded."""
        raise NotImplementedError

    @abc.abstractmethod
    def record(self, row: Dict[str, Any]) -> None:
        """Record the full result of a completed trial."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_all(self) -> pd.DataFrame:
        """Load all recorded trials into a DataFrame."""
        raise NotImplementedError


class CsvStorage(ExperimentStorage):
    """Adapter that persists trials to an append-only CSV file.
    
    Caches the existing keys in memory to make `should_run` O(1) without
    re-reading the file.
    """

    def __init__(self, csv_path: Path | str, key_columns: List[str], fieldnames: List[str]):
        self.csv_path = Path(csv_path)
        self.key_columns = list(key_columns)
        self.fieldnames = list(fieldnames)
        self._existing_keys: Optional[Set[Tuple]] = None

    def _load_existing_keys(self) -> Set[Tuple]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return set()
        df = pd.read_csv(self.csv_path)
        if len(df) == 0:
            return set()
        # Ensure we only extract the columns we care about, in the exact order
        try:
            records = df[self.key_columns].to_records(index=False)
            return set(tuple(r) for r in records)
        except KeyError:
            return set()

    def should_run(self, key_dict: Dict[str, Any]) -> bool:
        if self._existing_keys is None:
            self._existing_keys = self._load_existing_keys()
        
        key_tuple = tuple(key_dict[k] for k in self.key_columns)
        return key_tuple not in self._existing_keys

    def record(self, row: Dict[str, Any]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        
        # update internal cache so subsequent should_run checks are fast
        if self._existing_keys is not None:
            key_tuple = tuple(row[k] for k in self.key_columns)
            self._existing_keys.add(key_tuple)

    def load_all(self) -> pd.DataFrame:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return pd.DataFrame(columns=self.fieldnames)
        return pd.read_csv(self.csv_path)
