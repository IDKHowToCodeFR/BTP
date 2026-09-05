"""Shared constants: the QWS attribute schema.

These are the 9 real QoS attribute columns present in the QWS Dataset v2.0
(Al-Masri & Mahmoud, 2007), confirmed against the actual distributed file
(see data/qws_loader.py for provenance and a parsing note about embedded
commas in the source file). COST_ATTRIBUTES are attributes where a lower
raw value is better; everything else is BENEFIT-type (higher raw value is
better). data/preprocessing.py uses this split to invert cost attributes
during normalization so that, downstream, every column is uniformly
"higher = better".

IMPORTANT CORRECTION relative to the original project brief: QWS has no
monetary "cost" (price) attribute. An earlier draft of the task-profile
table (paper §3.7) listed "Cost" as a dominant attribute for two profiles
(Batch processing, Low-cost prototype). Since no such field exists in the
real dataset and fabricating one would mean scoring against a synthetic
attribute dressed up as real QoS data, those two profiles were redefined
to use only attributes that actually exist in QWS (see
tasks/profiles.py and baselines/lookup_table.py for the corrected
definitions, and the README "Corrections vs. the original brief" section
for the full rationale).
"""
from __future__ import annotations

QWS_ATTRIBUTE_COLUMNS = [
    "response_time",
    "availability",
    "throughput",
    "successability",
    "reliability",
    "compliance",
    "best_practices",
    "latency",
    "documentation",
]

COST_ATTRIBUTES = {"response_time", "latency"}
BENEFIT_ATTRIBUTES = [c for c in QWS_ATTRIBUTE_COLUMNS if c not in COST_ATTRIBUTES]

SERVICE_ID_COL = "service_id"

assert set(COST_ATTRIBUTES) | set(BENEFIT_ATTRIBUTES) == set(QWS_ATTRIBUTE_COLUMNS)
assert len(QWS_ATTRIBUTE_COLUMNS) == 9
