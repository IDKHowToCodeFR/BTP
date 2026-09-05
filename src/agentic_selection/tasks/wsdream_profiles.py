"""WS-DREAM-scoped task profiles for the supplementary validation pass
(paper §5.4). Reuses the exact same task-description text as the
corresponding QWS profiles in tasks/profiles.py -- see
baselines/wsdream_lookup_table.py's module docstring for why that reuse
is deliberate rather than incidental.
"""
from __future__ import annotations

from typing import List

from agentic_selection.tasks.profiles import TASK_PROFILE_BY_KEY, TaskProfile

WSDREAM_TASK_PROFILES: List[TaskProfile] = [
    TaskProfile(
        key="streaming",
        name=TASK_PROFILE_BY_KEY["streaming"].name,
        description=TASK_PROFILE_BY_KEY["streaming"].description,
        dominant_attributes=["throughput", "response_time"],
    ),
    TaskProfile(
        key="iot_telemetry_ingestion",
        name=TASK_PROFILE_BY_KEY["iot_telemetry_ingestion"].name,
        description=TASK_PROFILE_BY_KEY["iot_telemetry_ingestion"].description,
        dominant_attributes=["response_time", "throughput"],
    ),
]

WSDREAM_TASK_PROFILE_BY_KEY = {p.key: p for p in WSDREAM_TASK_PROFILES}
