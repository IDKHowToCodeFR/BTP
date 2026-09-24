# Architecture Overview

This project implements an **Agentic Cloud Service Selection** framework, comparing classical Multi-Criteria Decision Making (MCDM) algorithms against Large Language Models (LLMs) used as autonomous ranking agents.

## Core Modules

```mermaid
flowchart TD
    subgraph DataPipeline [Data Pipeline]
        D1["QWS / WS-DREAM Data"] --> D2["Normalize 0.0 - 1.0"]
        D2 --> D3["Candidate Pool"]
    end

    subgraph AgenticLoop [Agentic Reasoning Loop OODA]
        D3 --> P["Perception Layer<br>(Converts to Text)"]
        P --> RAG["RAG Agent Controller<br>(Procedural Gen & TF-IDF)"]
        RAG --> R["Reasoning Layer<br>(Prompt Construction)"]
        R <--> M["Memory<br>(Past Decisions)"]
        R --> LLM((LLM Backend))
        LLM --> R
        R --> C["Controller<br>(Dispatch Strategy)"]
    end

    subgraph MCDM [MCDM Evaluators]
        C -->|Strategy| S1("Weighted Sum")
        C -->|Strategy| S2("TOPSIS")
        C -->|Strategy| S3("Skyline + TOPSIS")
    end

    S1 --> OUT["Final Ranked Services"]
    S2 --> OUT
    S3 --> OUT
    
    style LLM fill:#ff9900,stroke:#333,stroke-width:2px,color:#000000
```

The architecture is heavily modularized to ensure LLM integrations are strictly decoupled from evaluation logic. 

### 1. Data Pipeline
- **`scripts/01_download_data.py` & `scripts/02_prepare_data.py`**: Fetches and prepares the QWS and WS-DREAM datasets. Data is normalized to a `[0, 1]` scale where `1.0` always represents the highest utility (e.g., lower latency = higher score).
- **`src/agentic_selection/evaluation/storage.py`**: Defines abstractions for incrementally appending results to local CSV files to support long-running, interruptible experiments.

### 2. The Agent Framework (`src/agentic_selection/agent/`)
The agent simulates an OODA (Observe, Orient, Decide, Act) loop across several decoupled files:
- **`llm_backends.py`**: Handles API communication. Currently supports Anthropic, OpenAI, and zero-cost local execution via `OllamaBackend`. Features robust exponential backoff retries to prevent connection drops during long evaluations. Tracks `prompt_tokens` and `completion_tokens`.
- **`perception.py`**: The "Observe" phase. Converts Pandas DataFrames into semantic markdown tables that the LLM can easily parse.
- **`reasoning.py`**: The "Orient" phase. Constructs the prompt and extracts structured JSON containing the LLM's inferred attribute weights, strategy, and justification.
- **`memory.py`**: The "Remember" phase. Stores historical decisions and justifications, allowing the agent to perform few-shot adaptation if drift is detected.
- **`controller.py`**: The orchestrator. Coordinates perception, reasoning, and memory, and dispatches the final ranking to the underlying MCDM functions based on the LLM's strategy choice.

### 3. RAG Agent Integration
A critical extension to the framework is the **`RAGAgentController`** (found in `controller.py`). It simulates Semantic Search / Retrieval-Augmented Generation for cloud selection:
- **Procedural Generation**: Because the original datasets only contain numeric data, this controller dynamically generates semantic text paragraphs for each service (e.g., converting a latency of `0.9` into `"excellent latency"`).
- **TF-IDF Retrieval**: Uses `scikit-learn`'s `TfidfVectorizer` to embed these generated documents and the natural language `task_description`. 
- **Filtering**: Computes cosine similarity and filters the candidate pool down to the Top-5 most semantically relevant services *before* passing them to the LLM. 
- **Impact**: This zero-cost procedural RAG pipeline drastically lowered the RAG agent's regret (from >0.30 to ~0.12), proving that accurate semantic retrieval significantly improves downstream reasoning.

### 4. Evaluation Protocols (`src/agentic_selection/evaluation/`)
- **`protocol.py`**: Defines `run_stable_protocol` (static context) and `run_drift_protocol` (dynamic context where optimal weights shift abruptly).
- **`metrics.py`**: Calculates Regret, Top-1 Accuracy, and Adaptation Lag.

### 5. Orchestration & Reporting
- **`run.bat`**: The central execution script that drives the entire pipeline end-to-end, safely appending data and bypassing already-completed stages.
- **`scripts/06_generate_report.py`**: Compiles CSV outputs into `report_tables.md`.
- **`scripts/10_generate_presentation_figures.py`**: Generates publication-ready `matplotlib` charts, applying strict aesthetic guidelines (`plot_style.py`), including the diverging `coolwarm` heatmaps for task-level regret.
