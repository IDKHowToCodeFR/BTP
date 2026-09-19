## Problem Statement

The system incurs tens of seconds of latency and API costs for every cloud service selection request because the LLM is invoked on every single request, even if the task description and environmental context are identical to a previous request.

## Solution

Implement an in-memory dictionary cache in the AgentController to instantly return previously inferred weights and strategies for identical (task, perception, memory) inputs, dropping latency to near-zero and skipping the LLM call entirely.

## User Stories

1. As a system operator, I want identical task descriptions to be resolved instantly without hitting the LLM API, so that I can reduce costs and latency.
2. As a system operator, I want the cache to be invalidated if the environmental context (perception text) changes, so that the agent can adapt to drift.
3. As a system operator, I want the cache to be invalidated if the memory context changes, so that the agent can learn from new past experiences.

## Implementation Decisions

- The `AgentController` will hold a simple Python dictionary `self._llm_cache` in memory.
- The cache key will be `(task_description, perception.to_prompt_text(), digest)`.
- If a cache hit occurs, we skip `get_agent_decision_raw` and use the cached parsed dict and raw text.
- We still record the decision in the JSONL memory log.

## Testing Decisions

- The agent controller's offline test suite already runs against mock LLMs; this change is internal to the controller and should keep all integration tests passing exactly as before while dropping API call counts for duplicate requests.

## Out of Scope

- Distributed caching (e.g., Redis). A simple instance-local dict cache is enough for the prototype scale.
- Vector-based semantic caching. Exact string matching is sufficient and far simpler (O(1) lookups).

## Further Notes

Ponytail: skipped semantic/vector DB overhead, added 5-line dict cache. Add vector caching only when exact-match hit rate is proven to be measurably insufficient.
