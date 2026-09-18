"""Agent controller (paper §3.1, §3.6): the perceive -> reason -> act ->
remember loop, executed once per selection request.

    PERCEIVE  : agent.perception.summarize_pool
    REASON    : agent.reasoning.get_agent_decision_raw + agent.validation
    ACT       : baselines.{weighted_sum,topsis,skyline_then_topsis}
    REMEMBER  : agent.memory.MemoryStore

This module intentionally contains no LLM-specific or dataset-specific
logic of its own -- it only orchestrates the four modules above, which is
what keeps each of them independently testable (see tests/).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agentic_selection.agent.llm_backends import LLMBackend
from agentic_selection.agent.memory import MemoryStore, format_digest, make_record
from agentic_selection.agent.perception import PoolPerception, summarize_pool
from agentic_selection.agent.reasoning import (
    DEFAULT_TOOL_MENU,
    LLMOutputParseError,
    get_agent_decision_raw,
)
from agentic_selection.agent.validation import ValidationResult, validate_agent_output
from agentic_selection.baselines import skyline_then_topsis, topsis, weighted_sum

ACTION_DISPATCH: Dict[str, Callable[[pd.DataFrame, dict, Sequence[str]], pd.Series]] = {
    "weighted_sum": weighted_sum,
    "topsis": topsis,
    "skyline_then_topsis": skyline_then_topsis,
}


@dataclass
class AgentDecision:
    record_id: str
    task_description: str
    perception: PoolPerception
    weights: Dict[str, float]
    strategy: str
    justification: str
    fallback_triggered: bool
    fallback_reason: Optional[str]
    ranking: pd.Series  # scores, best-first is NOT guaranteed here; see .top()
    latency_seconds: float
    api_calls: int
    raw_llm_text: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def top(self, k: int = 1) -> pd.Index:
        return self.ranking.sort_values(ascending=False).index[:k]

    def top_service_id(self) -> object:
        return self.top(1)[0]


class AgentController:
    """Stateful across calls only via its MemoryStore -- everything else
    (backend, attribute schema, tool menu) is fixed at construction time.
    """

    def __init__(
        self,
        backend: LLMBackend,
        attribute_cols: Sequence[str],
        memory_path,
        tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
        k_memory: int = 3,
    ):
        self.backend = backend
        self.attribute_cols = list(attribute_cols)
        self.memory = MemoryStore(memory_path)
        self.tool_menu = tuple(tool_menu)
        self.k_memory = k_memory
        self._llm_cache = {}

    def decide(
        self,
        task_description: str,
        candidate_pool: pd.DataFrame,
        strategy_override: Optional[str] = None,
        use_memory: bool = True,
    ) -> AgentDecision:
        """Run one full perceive-reason-act-remember cycle.

        Parameters
        ----------
        strategy_override : str, optional
            If given, force this strategy regardless of what the LLM (or
            the fallback) chose, while still using the LLM-inferred
            weights. This implements the "Agent (weights only)" ablation
            condition in paper Table 4.2 -- pass strategy_override="topsis"
            to reproduce that row; leave it None for the "Agent (full)" row.
        use_memory : bool
            If False, skip both memory retrieval and logging this
            decision to memory -- useful for a memory-free ablation to
            isolate memory's marginal contribution (paper §6, "Discuss
            ... whether memory-informed decisions measurably outperformed
            memory-free ones").
        """
        perception = summarize_pool(candidate_pool, self.attribute_cols)

        digest = ""
        if use_memory:
            neighbors = self.memory.nearest(perception.to_vector(), k=self.k_memory, query_task=task_description)
            digest = format_digest(neighbors)

        t0 = time.time()
        api_calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        cache_key = (task_description, perception.to_prompt_text(), digest)
        if cache_key in self._llm_cache:
            parsed, raw_text, usage_dict = self._llm_cache[cache_key]
            prompt_tokens = usage_dict.get("prompt_tokens", 0)
            completion_tokens = usage_dict.get("completion_tokens", 0)
        else:
            try:
                parsed, raw_text, usage_dict = get_agent_decision_raw(
                    self.backend, task_description, perception, self.attribute_cols, digest, self.tool_menu
                )
                api_calls = 1
                prompt_tokens = usage_dict.get("prompt_tokens", 0)
                completion_tokens = usage_dict.get("completion_tokens", 0)
                self._llm_cache[cache_key] = (parsed, raw_text, usage_dict)
            except LLMOutputParseError as e:
                parsed, raw_text = None, e.raw_text
                api_calls = 1
                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0}
                self._llm_cache[cache_key] = (parsed, raw_text, usage_dict)
        latency = time.time() - t0

        validated: ValidationResult = validate_agent_output(
            parsed, task_description, self.attribute_cols, self.tool_menu
        )

        effective_strategy = strategy_override or validated.strategy
        action_fn = ACTION_DISPATCH.get(effective_strategy)
        if action_fn is None:
            raise ValueError(
                f"strategy '{effective_strategy}' is not in ACTION_DISPATCH "
                f"{list(ACTION_DISPATCH)}"
            )
        ranking = action_fn(candidate_pool, validated.weights, self.attribute_cols)

        record = make_record(
            task_description=task_description,
            perception_vector=perception.to_vector(),
            weights=validated.weights,
            strategy=validated.strategy,  # log what the agent *chose*, even under override
            justification=validated.justification,
            fallback_triggered=validated.fallback_triggered,
            fallback_reason=validated.fallback_reason,
        )
        if use_memory:
            self.memory.append(record)

        return AgentDecision(
            record_id=record.record_id,
            task_description=task_description,
            perception=perception,
            weights=validated.weights,
            strategy=effective_strategy,
            justification=validated.justification,
            fallback_triggered=validated.fallback_triggered,
            fallback_reason=validated.fallback_reason,
            ranking=ranking,
            latency_seconds=latency,
            api_calls=api_calls,
            raw_llm_text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class RAGAgentController(AgentController):
    """A RAG (Retrieval-Augmented Generation) agent that simulates semantic search
    over service documentation before passing a filtered subset to the LLM.
    """
    
    def decide(
        self,
        task_description: str,
        candidate_pool: pd.DataFrame,
        strategy_override: Optional[str] = None,
        use_memory: bool = True,
    ) -> AgentDecision:
        # 1. Simulate Semantic Retrieval via Procedural Document Generation
        # Convert numerical QoS into text descriptions
        documents = []
        for service_id, row in candidate_pool.iterrows():
            desc_parts = [f"Service {service_id}"]
            for col in self.attribute_cols:
                val = row[col]
                # In normalized data, 1.0 is best. 
                if val > 0.8: qual = "excellent"
                elif val > 0.6: qual = "good"
                elif val > 0.4: qual = "average"
                elif val > 0.2: qual = "below average"
                else: qual = "poor"
                
                # Format feature name: "response_time" -> "response time"
                feature = col.replace("_", " ")
                desc_parts.append(f"{qual} {feature}")
            documents.append(". ".join(desc_parts) + ".")
            
        # 2. TF-IDF Retrieval
        vectorizer = TfidfVectorizer(stop_words='english')
        doc_vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([task_description])
        
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        top_k = min(5, len(candidate_pool))
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        filtered_pool = candidate_pool.iloc[top_indices].copy()
        
        # 2. Base Agent Decision on the retrieved subset
        decision = super().decide(task_description, filtered_pool, strategy_override, use_memory)
        
        # 3. Re-integrate into full pool ranking
        # Any service that wasn't retrieved gets a score of 0
        padded_ranking = pd.Series(0.0, index=candidate_pool.index)
        padded_ranking.update(decision.ranking)
        decision.ranking = padded_ranking
        
        return decision
