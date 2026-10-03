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
PROMPT_VERSION = "2.0"
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
        
        # Lenient repair: if flat weights are returned at root, wrap them in the expected envelope.
        # Check if the keys are primarily QoS attributes (like response_time, latency)
        qos_keywords = {"response_time", "availability", "throughput", "successability", "reliability", "compliance", "best_practices", "latency", "documentation"}
        if "weights" not in parsed and any(k in qos_keywords for k in parsed.keys()):
            parsed = {
                "weights": parsed,
                "strategy": "topsis",  # fallback default strategy if missing
                "justification": "auto-repaired flat JSON"
            }
        
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
        
    def get_schema(self, attribute_cols: Sequence[str], tool_menu: Sequence[str]) -> dict:
        ...


class DirectWeightReasoner:
    """The original regression reasoning strategy: forces the LLM to output
    continuous weights summing to 1.
    """
    def get_schema(self, attribute_cols: Sequence[str], tool_menu: Sequence[str]) -> dict:
        weight_props = {
            c: {"type": "number", "minimum": 0.0, "maximum": 1.0}
            for c in attribute_cols
        }
        return {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "properties": weight_props,
                    "required": list(attribute_cols),
                    "additionalProperties": False
                },
                "strategy": {
                    "type": "string",
                    "enum": list(tool_menu)
                },
                "justification": {"type": "string", "maxLength": 20}
            },
            "required": ["weights", "strategy", "justification"],
            "additionalProperties": False
        }
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
        
        json_schema = self.get_schema(attribute_cols, tool_menu)
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
    def __init__(self, categories_shuffle_seed=None):
        self.categories_shuffle_seed = categories_shuffle_seed

    @property
    def id(self) -> str:
        if self.categories_shuffle_seed is not None:
            return f"ClassificationReasoner_seed{self.categories_shuffle_seed}"
        return "ClassificationReasoner"

    def get_schema(self, attribute_cols: Sequence[str], tool_menu: Sequence[str]) -> dict:
        return {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "string"
                },
                "strategy": {
                    "type": "string",
                    "enum": list(tool_menu)
                },
                "justification": {"type": "string", "maxLength": 20}
            },
            "required": ["choice", "strategy", "justification"],
            "additionalProperties": False
        }
    def decide(
        self,
        backend: LLMBackend,
        task_description: str,
        perception: PoolPerception,
        attribute_cols: Sequence[str],
        memory_digest: str = "",
        tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
    ) -> Tuple[dict, str, dict]:
        from agentic_selection.baselines.lookup_table import TASK_LOOKUP_TABLE, GLOBAL_FIXED_WEIGHTS
        import json
        
        candidates_keys = list(TASK_LOOKUP_TABLE.keys())
        from difflib import get_close_matches
        nearest = get_close_matches(task_description, candidates_keys, n=2, cutoff=0.0)
        
        options = {"global_fixed": "Global fixed generic weights"}
        for k in set(nearest + [candidates_keys[0]]):
            if len(options) < 4:
                options[k] = f"Profile: {k}"
        
        cats = [f'- "{k}": {v}' for k, v in options.items()]
        categories = "\n".join(cats)
        
        system_prompt = CLASSIFICATION_SYSTEM_PROMPT.format(categories=categories)
        system_prompt = system_prompt.replace('"category": "<category_name>"', '"choice": "<id>"')
        system_prompt = system_prompt.replace('Category from list', 'choice from list')

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

        prompt = f"Task: {task_description}\n\n"
        if memory_section:
            prompt += memory_section
        prompt += f"Pool (N={perception.n_candidates}):\n{perception.to_prompt_text()}\n"
        
        schema = self.get_schema(attribute_cols, tool_menu)
        schema["properties"]["choice"]["enum"] = list(options.keys())

        raw_text, usage_dict = backend.complete(system_prompt, prompt, json_schema=schema)
        try:
            from agentic_selection.agent.reasoning import parse_llm_json, LLMOutputParseError
            parsed = parse_llm_json(raw_text)
        except Exception:
            parsed = {"choice": nearest[0] if nearest else "global_fixed", "strategy": "topsis", "justification": "parse fail"}

        choice = parsed.get("choice", "global_fixed")
        if choice == "global_fixed":
            weights = GLOBAL_FIXED_WEIGHTS
        else:
            weights = TASK_LOOKUP_TABLE.get(choice, GLOBAL_FIXED_WEIGHTS)

        final_dict = {
            "weights": weights,
            "strategy": parsed.get("strategy", ""),
            "justification": parsed.get("justification", ""),
            "category": choice,
        }
        return final_dict, raw_text, usage_dict

class VotingClassificationReasoner:
    """Invokes ClassificationReasoner 3 times with different option orders and takes a majority vote."""
    def __init__(self, n_votes: int = 3):
        self.n_votes = n_votes

    @property
    def id(self) -> str:
        return f"VotingClassificationReasoner_n{self.n_votes}"

    def get_schema(self, attribute_cols: Sequence[str], tool_menu: Sequence[str]) -> dict:
        return ClassificationReasoner().get_schema(attribute_cols, tool_menu)

    def decide(
        self,
        backend: LLMBackend,
        task_description: str,
        perception: PoolPerception,
        attribute_cols: Sequence[str],
        memory_digest: str = "",
        tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
    ) -> Tuple[dict, str, dict]:
        from collections import Counter
        
        votes = []
        raw_texts = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_duration": 0}

        for i in range(self.n_votes):
            # Pass a different seed so order changes each time
            r = ClassificationReasoner(categories_shuffle_seed=i)
            parsed, raw, usage = r.decide(backend, task_description, perception, attribute_cols, memory_digest, tool_menu)
            votes.append(parsed)
            raw_texts.append(raw)
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
                
        cats = [v.get("category") for v in votes if v and "category" in v]
        if not cats:
            raise LLMOutputParseError("All 3 classification votes failed to parse", "\n---\n".join(raw_texts))
            
        best_cat = Counter(cats).most_common(1)[0][0]
        best_vote = next((v for v in votes if v.get("category") == best_cat), votes[0])
        return best_vote, "\n---\n".join(raw_texts), total_usage
