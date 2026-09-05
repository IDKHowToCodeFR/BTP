from agentic_selection.agent.demo_backend import DemoHeuristicBackend, infer_demo_decision, infer_demo_profile
from agentic_selection.agent.reasoning import parse_llm_json
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS


def test_infer_demo_profile_detects_streaming_language():
    assert infer_demo_profile("live video stream with low latency") == "streaming"


def test_infer_demo_decision_covers_all_attributes_and_sums_to_one():
    decision = infer_demo_decision("banking payment backend")
    assert set(decision["weights"]) == set(QWS_ATTRIBUTE_COLUMNS)
    assert abs(sum(decision["weights"].values()) - 1.0) < 1e-9
    assert decision["strategy"] in {"weighted_sum", "topsis", "skyline_then_topsis"}


def test_demo_backend_returns_parseable_json_contract():
    backend = DemoHeuristicBackend()
    raw = backend.complete("", 'Task description:\n"""sensor telemetry ingestion"""')
    parsed = parse_llm_json(raw)
    assert set(parsed) == {"weights", "strategy", "justification"}
