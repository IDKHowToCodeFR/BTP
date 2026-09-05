"""Offline demo backend for the Streamlit showcase.

This backend is deliberately not used for reportable experiment results.
It returns structured, task-aware JSON quickly so the demo UI can show the
agent loop on machines without Ollama or API keys.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Sequence

from agentic_selection.agent.llm_backends import LLMBackend
from agentic_selection.baselines.lookup_table import TASK_LOOKUP_TABLE
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS


PROFILE_KEYWORDS = {
    "streaming": ("stream", "video", "audio", "latency", "live", "sports", "notification"),
    "financial_transaction": ("bank", "payment", "payroll", "transaction", "fraud", "money"),
    "batch_processing": ("batch", "overnight", "cron", "catalog", "re-index", "million"),
    "low_cost_prototype": ("prototype", "hobby", "student", "hackathon", "documentation", "demo"),
    "iot_telemetry_ingestion": ("iot", "sensor", "telemetry", "thermostat", "device"),
    "compliance_sensitive_backend": ("compliance", "audit", "regulated", "standards", "conformance"),
}


def infer_demo_profile(task_description: str) -> str:
    """Return the closest profile key using transparent keyword scoring."""
    text = task_description.lower()
    best_key = "streaming"
    best_score = -1
    for key, words in PROFILE_KEYWORDS.items():
        score = sum(1 for word in words if word in text)
        if score > best_score:
            best_key = key
            best_score = score
    return best_key


def _renormalize_to_attributes(weights: Dict[str, float], attribute_cols: Sequence[str]) -> Dict[str, float]:
    out = {attr: float(weights.get(attr, 0.0)) for attr in attribute_cols}
    total = sum(out.values())
    if total <= 0:
        return {attr: 1.0 / len(attribute_cols) for attr in attribute_cols}
    return {attr: value / total for attr, value in out.items()}


def infer_demo_decision(task_description: str, attribute_cols: Sequence[str] = QWS_ATTRIBUTE_COLUMNS) -> dict:
    """Create a plausible controller response for demos and tests.

    The result follows the same JSON contract a real LLM backend must
    satisfy: weights, strategy, and justification.
    """
    profile = infer_demo_profile(task_description)
    weights = _renormalize_to_attributes(TASK_LOOKUP_TABLE[profile], attribute_cols)

    text = task_description.lower()
    if any(word in text for word in ("audit", "compliance", "regulated", "hard", "non-negotiable")):
        strategy = "skyline_then_topsis"
    elif any(word in text for word in ("stream", "live", "telemetry", "fast", "latency")):
        strategy = "weighted_sum"
    else:
        strategy = "topsis"

    return {
        "weights": weights,
        "strategy": strategy,
        "justification": (
            f"Offline demo reasoning mapped the request to the {profile} profile, "
            f"weighted its dominant QoS attributes, and selected {strategy} for a fast, explainable demonstration."
        ),
    }


class DemoHeuristicBackend(LLMBackend):
    """LLMBackend-compatible offline responder for interactive demos."""

    def __init__(self, attribute_cols: Sequence[str] = QWS_ATTRIBUTE_COLUMNS):
        self.attribute_cols = list(attribute_cols)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        match = re.search(r'Task description:\s*"""(.*?)"""', user_prompt, flags=re.DOTALL)
        task_description = match.group(1).strip() if match else user_prompt
        return json.dumps(infer_demo_decision(task_description, self.attribute_cols))
