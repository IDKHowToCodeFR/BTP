import time
import pandas as pd
from agentic_selection.agent.controller import AgentController
from agentic_selection.agent.llm_backends import build_backend_from_config
from agentic_selection.tasks.profiles import TASK_PROFILES

import yaml

def main():
    print("Initializing AgentController (loading LLM configured in config.yaml)...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    backend = build_backend_from_config(config["llm"])
    
    # Use standard 9 attributes
    attributes = ["response_time", "availability", "throughput", "successability", 
                  "reliability", "compliance", "best_practices", "latency", "documentation"]
    
    controller = AgentController(
        backend=backend,
        attribute_cols=attributes,
        memory_path="memory_store/agent_memory.jsonl",
        k_memory=3
    )

    # Make a dummy candidate pool (3 rows)
    dummy_pool = pd.DataFrame([
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ], columns=attributes)
    
    task_desc = TASK_PROFILES[0].description
    
    print(f"\n--- Run 1: First request (Cache Miss) ---")
    print(f"Task: {task_desc[:60]}...")
    t0 = time.time()
    res1 = controller.decide(task_desc, dummy_pool)
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
    print(f"API calls made: {res1.api_calls}")
    print(f"RAG Few-Shot Active? {'Yes' if res1.raw_llm_text and 'Examples of past similar tasks' in res1.raw_llm_text else 'No'}")
    
    print(f"\n--- Run 2: Identical request (Cache Hit) ---")
    t0 = time.time()
    res2 = controller.decide(task_desc, dummy_pool)
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.4f} seconds")
    print(f"API calls made: {res2.api_calls}")

if __name__ == "__main__":
    main()
