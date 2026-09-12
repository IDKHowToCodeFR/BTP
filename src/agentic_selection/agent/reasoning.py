"""Reasoning module (paper §3.6): builds the prompt shown to the LLM and
parses its structured JSON response.

The LLM's job is scoped narrowly and deliberately: infer attribute
weights and pick a strategy from a fixed menu, and explain why. It never
performs the ranking arithmetic itself -- that stays in baselines/,
deterministic and auditable (see paper §1.3, §3.6).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence, Tuple

from agentic_selection.agent.llm_backends import LLMBackend
from agentic_selection.agent.perception import PoolPerception

DEFAULT_TOOL_MENU: Tuple[str, ...] = ("weighted_sum", "topsis", "skyline_then_topsis")


class LLMOutputParseError(Exception):
    """Raised when the model's response cannot be parsed into the
    expected {weights, strategy, justification} JSON object. Callers
    (agent/controller.py) treat this exactly like a validation failure --
    it triggers the lookup-table fallback described in paper §3.6 -- but
    it is kept as a distinct exception type from validation.py's semantic
    failures (e.g. weights not summing to 1) so the two failure modes can
    be logged and counted separately if desired.
    """

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


SYSTEM_PROMPT_TEMPLATE = """You are the reasoning component of an agentic cloud-service-selection system.

Your ONLY job is to decide two things for the current request:
1. How much weight (0 to 1, summing to 1 across all listed attributes) each QoS attribute should get, given the natural-language task description.
2. Which ONE selection strategy from the fixed menu below best fits the current candidate pool's characteristics.

You do NOT rank or score the candidate services yourself -- that is done afterward by deterministic code using the weights and strategy you provide. Do not invent attributes that are not in the provided list. Do not invent strategies that are not in the provided menu.

Available QoS attributes (weights must cover exactly these, nothing else): {attribute_list}

Available strategies (pick exactly one):
- "weighted_sum": simple linear combination of weighted attributes. Robust, transparent, works well when attributes don't trade off sharply against each other.
- "topsis": ranks by geometric closeness to an ideal point and distance from a worst-case point. Tends to reward well-rounded candidates over candidates that are extreme on one attribute; often preferable when several near-dominant candidates make a simple sum hard to discriminate between.
- "skyline_then_topsis": first filters to only the Pareto-optimal (non-dominated) candidates, then ranks that smaller set with TOPSIS. Useful when the pool is large and you want to first discard anything that is unambiguously worse than some other option on every attribute at once, before doing finer-grained ranking.

Respond with ONLY a single JSON object and nothing else -- no markdown code fences, no preamble, no explanation outside the JSON. The JSON object must have exactly these three keys:
{{
  "weights": {{"attribute_name": <float>, ...}},
  "strategy": "<one of the strategy names above>",
  "justification": "<one or two sentences explaining the weight and strategy choice in terms of the task description and pool characteristics>"
}}
"""

USER_PROMPT_TEMPLATE = """Task description:
\"\"\"{task_description}\"\"\"

Current candidate pool characteristics:
{perception_text}

{memory_digest_section}Respond now with only the JSON object described in the system prompt."""


def build_prompt(
    task_description: str,
    perception: PoolPerception,
    attribute_cols: Sequence[str],
    memory_digest: str = "",
    tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
) -> Tuple[str, str]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        attribute_list=", ".join(attribute_cols),
    )
    if set(tool_menu) != set(DEFAULT_TOOL_MENU):
        # Only reachable if a caller customizes the tool menu; keep the
        # system prompt in sync rather than silently describing tools
        # that aren't actually on offer.
        lines = "\n".join(f'- "{t}"' for t in tool_menu)
        system_prompt = re.sub(
            r"Available strategies.*?Respond with ONLY",
            f"Available strategies (pick exactly one):\n{lines}\n\nRespond with ONLY",
            system_prompt,
            flags=re.DOTALL,
        )

    memory_section = ""
    if memory_digest.strip():
        memory_section = f"Examples of past similar tasks and valid JSON outputs (use these as a reference for formatting and reasoning):\n{memory_digest}\n\n"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        task_description=task_description,
        perception_text=perception.to_prompt_text(),
        memory_digest_section=memory_section,
    )
    return system_prompt, user_prompt


def parse_llm_json(raw_text: str) -> dict:
    """Parse the model's response into a dict, tolerating the common ways
    real models fail to follow a "JSON only" instruction exactly (wrapping
    in markdown fences, adding a leading/trailing sentence). Raises
    LLMOutputParseError, carrying the raw text, if no valid JSON object
    with the right shape can be extracted at all.
    """
    text = raw_text.strip()

    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        candidates.insert(0, fence_match.group(1))
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = e
            continue
        if not isinstance(parsed, dict):
            last_error = ValueError(f"parsed JSON is not an object: {type(parsed)}")
            continue
        return parsed

    raise LLMOutputParseError(
        f"Could not parse a JSON object from the model's response "
        f"(last error: {last_error!r})",
        raw_text=raw_text,
    )


def get_agent_decision_raw(
    backend: LLMBackend,
    task_description: str,
    perception: PoolPerception,
    attribute_cols: Sequence[str],
    memory_digest: str = "",
    tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
) -> Tuple[dict, str, dict]:
    """Build the prompt, call the backend, and parse its response.

    Returns (parsed_dict, raw_text, usage_dict). Raises LLMOutputParseError if the
    response can't be parsed -- callers should catch this and route into
    the validation/fallback path (agent/controller.py does this).
    """
    system_prompt, user_prompt = build_prompt(
        task_description, perception, attribute_cols, memory_digest, tool_menu
    )
    raw_text, usage_dict = backend.complete(system_prompt, user_prompt)
    parsed = parse_llm_json(raw_text)
    return parsed, raw_text, usage_dict
