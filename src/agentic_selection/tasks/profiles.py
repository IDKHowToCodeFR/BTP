"""The six task profiles used throughout the evaluation, plus a held-out
set of paraphrased/novel task descriptions used to test generalization
beyond the lookup table's fixed coverage (paper §3.7, §4.2, hypothesis H3).

Each profile has:
- key: matches a key in baselines.lookup_table.TASK_LOOKUP_TABLE
- name: human-readable label
- description: the canonical natural-language task description shown to
  both the lookup table's fuzzy matcher and the LLM reasoning module
- dominant_attributes: which QWS attributes should get most of the
  weight, purely for documentation/sanity-checking -- the agent is never
  shown this list directly, only `description`.

HELD_OUT_TASKS are deliberately *not* one of the six canonical
descriptions verbatim: they either paraphrase a profile heavily, combine
two profiles' concerns, or describe a scenario not covered by any profile
at all. get_lookup_weights()'s fuzzy string matching is expected to
struggle with most of these (that gap is precisely what H3 measures), while
an LLM reading the actual sentence should not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TaskProfile:
    key: str
    name: str
    description: str
    dominant_attributes: List[str] = field(default_factory=list)


TASK_PROFILES: List[TaskProfile] = [
    TaskProfile(
        key="streaming",
        name="Streaming",
        description=(
            "We need a service to power real-time video and audio delivery to "
            "end users. Sustained throughput and low end-to-end latency matter "
            "far more than anything else -- a brief buffering stall is "
            "immediately visible to viewers, but a rare failed request that "
            "gets silently retried is not a big deal."
        ),
        dominant_attributes=["throughput", "response_time", "latency"],
    ),
    TaskProfile(
        key="financial_transaction",
        name="Financial transaction",
        description=(
            "This service will sit behind a banking/payments backend that "
            "processes real money transfers. It must be up when customers need "
            "it and it must not silently drop or corrupt a transaction -- "
            "availability and reliability are non-negotiable, even if that "
            "means somewhat higher response times."
        ),
        dominant_attributes=["reliability", "availability"],
    ),
    TaskProfile(
        key="batch_processing",
        name="Batch processing",
        description=(
            "We're running large, unattended overnight data-processing jobs. "
            "Nobody is watching the job run, so it needs to actually finish "
            "without failing partway through, and it needs to chew through a "
            "large volume of records efficiently. Response time for any single "
            "call barely matters since nothing is waiting on it interactively."
        ),
        dominant_attributes=["throughput", "successability"],
    ),
    TaskProfile(
        key="low_cost_prototype",
        name="Low-cost prototype",
        description=(
            "This is an early-stage hobby project with no dedicated ops team "
            "and no budget for support contracts. Whatever we pick needs to be "
            "well documented and easy to integrate correctly on the first try, "
            "since we can't afford to spend days debugging an unfamiliar API "
            "with poor docs."
        ),
        dominant_attributes=["documentation", "best_practices"],
    ),
    TaskProfile(
        key="iot_telemetry_ingestion",
        name="IoT telemetry ingestion",
        description=(
            "We have thousands of small sensor devices each sending tiny "
            "telemetry messages many times per minute. Each individual call "
            "needs to complete quickly and needs to actually succeed, since a "
            "failed telemetry push usually just means that data point is lost "
            "forever rather than retried."
        ),
        dominant_attributes=["response_time", "successability"],
    ),
    TaskProfile(
        key="compliance_sensitive_backend",
        name="Compliance-sensitive backend",
        description=(
            "This service will be used in a regulated industry where we will "
            "be audited on standards conformance. It needs to demonstrably "
            "follow established web service best practices and compliance "
            "requirements -- an auditor rejecting our vendor choice is a "
            "bigger problem for us than a slightly slower response time."
        ),
        dominant_attributes=["compliance", "best_practices"],
    ),
]

TASK_PROFILE_BY_KEY = {p.key: p for p in TASK_PROFILES}


@dataclass(frozen=True)
class HeldOutTask:
    description: str
    nearest_profile_key: str
    note: str


# Held-out set for H3 (generalization to unseen phrasing). `nearest_profile_key`
# records which canonical profile a human would say this is *closest* to, so
# the evaluation harness can still compute regret against a reference weight
# vector -- but the description text itself is deliberately not what the
# lookup table's fuzzy matcher was tuned against.
HELD_OUT_TASKS: List[HeldOutTask] = [
    HeldOutTask(
        description=(
            "Picking a backend for a live sports-score push notification "
            "feed -- fans should see a goal within a second or two of it "
            "happening, or the notification is basically worthless."
        ),
        nearest_profile_key="streaming",
        note="Paraphrase of the streaming/low-latency concern in a different domain.",
    ),
    HeldOutTask(
        description=(
            "This will handle payroll disbursement for a mid-size company. "
            "If a payment silently fails or the service is down on payday, "
            "that's a serious problem for a lot of people at once."
        ),
        nearest_profile_key="financial_transaction",
        note="Domain-shifted paraphrase of the financial-transaction profile.",
    ),
    HeldOutTask(
        description=(
            "We need to re-index our entire product catalog against a "
            "third-party enrichment API once a week, several million SKUs "
            "in one run, kicked off by a cron job with nobody watching it."
        ),
        nearest_profile_key="batch_processing",
        note="Paraphrase of batch processing with different domain vocabulary.",
    ),
    HeldOutTask(
        description=(
            "A student side-project with zero budget -- whatever we choose "
            "has to have clear examples we can copy-paste from, because "
            "none of us has time to read a spec to figure out the request "
            "format."
        ),
        nearest_profile_key="low_cost_prototype",
        note="Paraphrase of the low-cost/hobby profile.",
    ),
    HeldOutTask(
        description=(
            "Selecting a service for a smart-thermostat fleet reporting "
            "temperature readings every few seconds; missing one reading "
            "occasionally is fine, but the request needs to return before "
            "the next reading is already due."
        ),
        nearest_profile_key="iot_telemetry_ingestion",
        note="Paraphrase of IoT ingestion using a different device type.",
    ),
    HeldOutTask(
        description=(
            "We're being audited under an industry standards framework "
            "next quarter and need every vendor in the stack to have a "
            "clean conformance story on paper, or we risk failing the audit."
        ),
        nearest_profile_key="compliance_sensitive_backend",
        note="Paraphrase of compliance-sensitive profile with audit framing.",
    ),
    HeldOutTask(
        description=(
            "We need one service that has to be both extremely reliable "
            "AND extremely fast, because it's a real-time fraud check that "
            "runs synchronously before a financial transaction is approved."
        ),
        nearest_profile_key="financial_transaction",
        note=(
            "Deliberate blend of two profiles' concerns (reliability + "
            "speed) not matching any single profile cleanly -- tests "
            "whether the agent can weight a genuine trade-off rather than "
            "pattern-matching to one canonical profile."
        ),
    ),
    HeldOutTask(
        description=(
            "Just need something that works for a weekend hackathon demo "
            "in front of a few judges; it only has to survive a 10-minute "
            "demo, nothing about it needs to be production-grade."
        ),
        nearest_profile_key="low_cost_prototype",
        note="Novel scenario with no close analogue among the six profiles.",
    ),
]


def all_task_keys() -> List[str]:
    return [p.key for p in TASK_PROFILES]
