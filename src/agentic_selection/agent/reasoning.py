"""Reasoning module. Builds prompts and parses structured JSON responses from LLMs."""
# ============================== #
#         Reasoning Module       #
# ============================== #

# --- Imports ---
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Sequence, Tuple, Protocol
from agentic_selection.agent.llm_backends import LLMBackend
from agentic_selection.agent.perception import PoolPerception
from agentic_selection.baselines.lookup_table import TASK_LOOKUP_TABLE, get_lookup_weights

# --- Globals & Errors ---
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


SYSTEM_PROMPT_TEMPLATE = """You = agent reasoner.

Decide 2 things:
1. QoS weights (0-1, sum to 1).
2. 1 strategy from menu.

DO NOT RANK. Output JSON only. No invented attrs/strategies.

Attrs (use exactly these): {attribute_list}

Strategies (pick 1):
- "weighted_sum": Linear. Good when no sharp trade-offs.
- "topsis": Geometric distance to ideal. Good for well-rounded candidates.
- "skyline_then_topsis": Pareto filter → TOPSIS. Good for large pools to drop dominated candidates first.

Output strictly JSON:
{{
  "weights": {{"attr": <float>, ...}},
  "strategy": "<strategy_name>",
  "justification": "<reason>"
}}
"""

USER_PROMPT_TEMPLATE = """Task:
\"\"\"{task_description}\"\"\"

Pool:
{perception_text}

{memory_digest_section}Output JSON now."""


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
            r"Strategies \(pick 1\):.*?Output strictly JSON:",
            f"Strategies (pick 1):\n{lines}\n\nOutput strictly JSON:",
            system_prompt,
            flags=re.DOTALL,
        )

    memory_section = ""
    if memory_digest.strip():
        memory_section = f"Past valid JSON examples:\n{memory_digest}\n\n"

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

    # Real-world LLMs often wrap JSON outputs in markdown blocks even when strictly prompted.
    # We must scan for fences or naked braces before raising a parsing error.
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


class ReasoningStrategy(Protocol):
    def decide(
        self,
        backend: LLMBackend,
        task_description: str,
        perception: PoolPerception,
        attribute_cols: Sequence[str],
        memory_digest: str = "",
        tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
    ) -> Tuple[dict, str, dict]:
        ...


class DirectWeightReasoner:
    """The original regression reasoning strategy: forces the LLM to output
    continuous weights summing to 1.
    """
    def decide(
        self,
        backend: LLMBackend,
        task_description: str,
        perception: PoolPerception,
        attribute_cols: Sequence[str],
        memory_digest: str = "",
        tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
    ) -> Tuple[dict, str, dict]:
        system_prompt, user_prompt = build_prompt(
            task_description, perception, attribute_cols, memory_digest, tool_menu
        )
        
        # We pass the default schema for DirectWeightReasoner
        json_schema = {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                },
                "strategy": {"type": "string"},
                "justification": {"type": "string"}
            },
            "required": ["weights", "strategy", "justification"]
        }
        raw_text, usage_dict = backend.complete(system_prompt, user_prompt, json_schema=json_schema)
        parsed = parse_llm_json(raw_text)
        return parsed, raw_text, usage_dict


CLASSIFICATION_SYSTEM_PROMPT = """You = agent reasoner.

Decide 2 things:
1. Category from list.
2. 1 strategy from menu.

DO NOT RANK. Output JSON only. No invented categories.

Categories (pick 1):
{categories}

Strategies (pick 1):
- "weighted_sum": Linear. Good when no sharp trade-offs.
- "topsis": Geometric distance to ideal. Good for well-rounded candidates.
- "skyline_then_topsis": Pareto filter → TOPSIS. Good for large pools to drop dominated candidates first.

Output strictly JSON:
{{
  "category": "<category_name>",
  "strategy": "<strategy_name>",
  "justification": "<reason>"
}}
"""

class ClassificationReasoner:
    """An SLM-optimized reasoning strategy that asks the LLM to classify the
    task into a known profile, then maps that profile to optimal weights.
    """
    def decide(
        self,
        backend: LLMBackend,
        task_description: str,
        perception: PoolPerception,
        attribute_cols: Sequence[str],
        memory_digest: str = "",
        tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
    ) -> Tuple[dict, str, dict]:
        categories = "\n".join(f"- {k}" for k in TASK_LOOKUP_TABLE.keys())
        system_prompt = CLASSIFICATION_SYSTEM_PROMPT.format(categories=categories)
        if set(tool_menu) != set(DEFAULT_TOOL_MENU):
            lines = "\n".join(f'- "{t}"' for t in tool_menu)
            system_prompt = re.sub(
                r"Strategies \(pick 1\):.*?Output strictly JSON:",
                f"Strategies (pick 1):\n{lines}\n\nOutput strictly JSON:",
                system_prompt,
                flags=re.DOTALL,
            )

        memory_section = ""
        if memory_digest.strip():
            memory_section = f"Past valid JSON examples:\n{memory_digest}\n\n"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            task_description=task_description,
            perception_text=perception.to_prompt_text(),
            memory_digest_section=memory_section,
        )

        json_schema = {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "strategy": {"type": "string"},
                "justification": {"type": "string"}
            },
            "required": ["category", "strategy", "justification"]
        }

        raw_text, usage_dict = backend.complete(system_prompt, user_prompt, json_schema=json_schema)
        parsed = parse_llm_json(raw_text)

        category = parsed.get("category", "")

        # If the LLM hallucinates a category not in TASK_LOOKUP_TABLE despite strict 
        # JSON schema enforcements, we fallback to the 'global' median weights to guarantee mathematical safety.
        weights = get_lookup_weights(category, fallback="global")

        final_dict = {
            "weights": weights,
            "strategy": parsed.get("strategy", ""),
            "justification": parsed.get("justification", ""),
            "category": category,
        }
        return final_dict, raw_text, usage_dict
