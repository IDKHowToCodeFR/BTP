"""Validation module (paper §3.6): a deterministic gate between whatever
the LLM returned and what actually gets executed. This is the mechanism
that bounds the damage of a hallucinated or malformed response -- its
trigger rate is itself an evaluation metric (paper §4.1, "fallback
trigger rate").

Failure modes that trigger the lookup-table fallback (any one is
sufficient):
  1. The response could not be parsed as JSON at all (LLMOutputParseError
     upstream in reasoning.py).
  2. "strategy" is missing, not a string, or not in the allowed tool menu.
  3. "weights" is missing, not a dict, contains a key outside the known
     attribute vocabulary, contains a negative value, or sums to
     (numerically) zero so it cannot be renormalized.
  4. The (renormalized) weights are "degenerate": a single attribute
     holds more than DEGENERATE_SINGLE_ATTR_THRESHOLD of the total mass.
     This threshold is deliberately generous (0.85) -- the most
     concentrated hand-authored profile in this project's own lookup
     table tops out around 0.32-0.34 on any single attribute, so 0.85 is
     not a realistic value for a genuinely task-appropriate weighting; it
     is a plausible value for a degenerate/hallucinated one (paper's own
     example: "all mass on one attribute for a task profile where that
     would be implausible").

Weights that merely fail to sum to 1.0 are NOT a failure -- they are
renormalized. Only the failure modes above trigger a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from agentic_selection.agent.reasoning import DEFAULT_TOOL_MENU
from agentic_selection.baselines.lookup_table import get_lookup_weights

DEGENERATE_SINGLE_ATTR_THRESHOLD = 0.85
DEFAULT_FALLBACK_STRATEGY = "topsis"


def default_degenerate_threshold(n_attributes: int) -> float:
    """The "is this weighting degenerate" bar has to depend on how many
    attributes there are to spread weight across. With 9 attributes
    (QWS), any single one holding >85% is implausible for a genuinely
    task-appropriate weighting -- that's DEGENERATE_SINGLE_ATTR_THRESHOLD,
    calibrated against this project's own 9-attribute lookup table, whose
    most concentrated profile tops out around 32-34% on any one attribute.
    But with only 2 attributes (e.g. WS-DREAM Dataset #1, which only has
    response_time and throughput -- see data/wsdream_loader.py), a
    legitimately reasoned "throughput matters far more here" weighting of
    e.g. 90/10 is completely ordinary, not a hallucination sign, and
    flagging it as degenerate would just be a false-positive fallback
    trigger inflating the reported fallback rate for no real reliability
    reason. This scales the bar with attribute count instead of using one
    fixed global constant.
    """
    if n_attributes <= 2:
        return 0.95
    if n_attributes <= 5:
        return 0.90
    return DEGENERATE_SINGLE_ATTR_THRESHOLD


@dataclass
class ValidationResult:
    weights: dict
    strategy: str
    justification: str
    fallback_triggered: bool
    fallback_reason: Optional[str]


def _fallback(
    task_description: str,
    reason: str,
    attribute_cols: Sequence[str],
    fallback_strategy: str = DEFAULT_FALLBACK_STRATEGY,
) -> ValidationResult:
    weights = get_lookup_weights(task_description, fallback="nearest")
    # Restrict/reorder to exactly attribute_cols in case the lookup table
    # was built for a different (e.g. larger) attribute schema than the
    # current candidate pool actually has.
    weights = {c: weights.get(c, 0.0) for c in attribute_cols}
    total = sum(weights.values())
    if total <= 0:
        weights = {c: 1.0 / len(attribute_cols) for c in attribute_cols}
    else:
        weights = {c: v / total for c, v in weights.items()}
    return ValidationResult(
        weights=weights,
        strategy=fallback_strategy,
        justification=f"[FALLBACK] {reason}",
        fallback_triggered=True,
        fallback_reason=reason,
    )


def validate_agent_output(
    parsed: Optional[dict],
    task_description: str,
    attribute_cols: Sequence[str],
    tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
    degenerate_threshold: Optional[float] = None,
) -> ValidationResult:
    """Validate (and, where possible, repair) a parsed LLM response.

    `parsed` may be None (upstream JSON parsing failed entirely) -- this
    is itself grounds for an immediate fallback.

    `degenerate_threshold` overrides the auto-derived
    `default_degenerate_threshold(len(attribute_cols))` if given
    explicitly; leave it None to get the attribute-count-aware default
    (see that function's docstring for why a fixed constant is wrong).
    """
    if parsed is None:
        return _fallback(task_description, "LLM response was not parseable JSON", attribute_cols)

    strategy = parsed.get("strategy")
    if not isinstance(strategy, str) or strategy not in tool_menu:
        return _fallback(
            task_description,
            f"strategy field missing or not in allowed menu {tuple(tool_menu)}: {strategy!r}",
            attribute_cols,
        )

    weights = parsed.get("weights")
    if not isinstance(weights, dict) or len(weights) == 0:
        return _fallback(task_description, "weights field missing or not a non-empty object", attribute_cols)

    known = set(attribute_cols)
    unknown_keys = set(weights.keys()) - known
    if unknown_keys:
        return _fallback(
            task_description,
            f"weights referenced unknown attribute(s) not in the pool's schema: {sorted(unknown_keys)}",
            attribute_cols,
        )

    try:
        numeric_weights = {k: float(v) for k, v in weights.items()}
    except (TypeError, ValueError):
        return _fallback(task_description, "weights contained a non-numeric value", attribute_cols)

    if any(v < 0 for v in numeric_weights.values()):
        return _fallback(task_description, "weights contained a negative value", attribute_cols)

    # Fill any attribute the model omitted with 0 -- an omission is not
    # itself a hard failure (a sparse-but-valid weighting is plausible),
    # but it does count toward the degeneracy check below.
    full_weights = {c: numeric_weights.get(c, 0.0) for c in attribute_cols}
    total = sum(full_weights.values())
    if total <= 1e-9:
        return _fallback(task_description, "weights summed to (numerically) zero", attribute_cols)

    normalized = {c: v / total for c, v in full_weights.items()}

    max_attr, max_weight = max(normalized.items(), key=lambda kv: kv[1])
    threshold = degenerate_threshold if degenerate_threshold is not None else default_degenerate_threshold(len(attribute_cols))
    if max_weight > threshold:
        return _fallback(
            task_description,
            f"degenerate weights: '{max_attr}' alone holds {max_weight:.2f} "
            f"of total mass (> {threshold} threshold for {len(attribute_cols)} attributes)",
            attribute_cols,
        )

    justification = parsed.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        justification = "(no justification provided by model)"

    return ValidationResult(
        weights=normalized,
        strategy=strategy,
        justification=justification,
        fallback_triggered=False,
        fallback_reason=None,
    )
