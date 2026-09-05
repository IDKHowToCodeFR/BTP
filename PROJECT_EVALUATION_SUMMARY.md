# Project Evaluation Summary

Date checked: 2026-09-04

## What Is Built

This repository implements an agentic cloud/web-service selection system. The LLM is used only as a controller: it infers QoS attribute weights from a natural-language task description and chooses among deterministic MCDM tools. The ranking mathematics remains auditable Python code:

- `weighted_sum`
- `topsis`
- `skyline`
- `skyline_then_topsis`

The loop is implemented as perceive, reason, act, remember:

- Perceive: summarize the candidate QoS pool.
- Reason: request a structured weight vector, strategy, and justification.
- Act: run the chosen deterministic ranking method.
- Remember: append the decision to a JSONL memory log for future retrieval.

## What Was Verified

The local validation run completed successfully:

```text
139 passed, 1 skipped
```

The skipped test is the optional real-file WS-DREAM Dataset #1 check because `data/raw/wsdream1/` is not present in this checkout. The real QWS file is present and validated:

```text
rows: 2507 | is_synthetic: False
all 9 QWS attribute columns present and within [0, 1] -- OK
no missing values in attribute columns -- OK
```

## Current Experiment Artifacts

The current checked result files are:

- `results/tables/stable_results.csv`: 168 rows
- `results/tables/drift_results.csv`: 24 rows
- `results/tables/report_tables.md`: regenerated report tables
- `results/tables/extended_analysis.md`: BTP-facing aggregate analysis
- `results/figures/regret_by_condition.png`
- `results/figures/adaptation_lag_by_condition.png`
- `results/figures/regret_by_task_kind.png`
- `results/figures/agent_latency_by_condition.png`
- `results/figures/strategy_selection_frequency.png`
- `results/figures/drift_trace_example.png`

The stable run is smoke-scale: 14 tasks, 3 candidate pools per task, 4 conditions. The drift run is smoke-scale: 6 task profiles, 1 drift trial per profile, 4 conditions.

## Main Findings

The task-aware lookup table is the strongest method under stable QWS conditions in this smoke run:

| condition | mean regret |
|---|---:|
| lookup_table | 0.0069 |
| global_fixed | 0.0108 |
| agent_full | 0.0350 |
| agent_weights_only | 0.0352 |

The agent is strongest under drift:

| condition | mean adaptation lag |
|---|---:|
| global_fixed | 2.0 rounds |
| lookup_table | 2.0 rounds |
| agent_weights_only | 0.0 rounds |
| agent_full | 0.0 rounds |

The validation/fallback rate in the stable run is 0.0 for both agent conditions. The operational cost is not negligible: the recorded stable run shows one model call per agent decision and mean latencies of 70.76 seconds for the weights-only agent and 40.76 seconds for the full agent.

## Hypothesis Status

H1, stable seen-task parity with the lookup table: not supported in the smoke run. The lookup table has lower regret.

H2, lower adaptation lag under drift: supported under the stated re-evaluation regime.

H3, better held-out task handling: contradicted in the smoke run. The agent's held-out mean regret is higher than the lookup table fallback.

H4, non-trivial cost: supported. Agent decisions incur one model call and tens of seconds of latency in the recorded run.

## How To Demonstrate

From the project root:

```bash
python -m pytest tests -q
python scripts/03_validate_baselines.py
python scripts/04_run_stable_experiment.py --n-pools 3 --yes
python scripts/05_run_drift_experiment.py --n-trials 1 --yes
python scripts/06_generate_report.py
python scripts/08_generate_extended_analysis.py
python -m streamlit run app/streamlit_demo.py --server.port 8501
```

The stable and drift experiment commands are resumable. With the current result CSVs present, the smoke commands verify the pipeline without recomputing existing LLM calls.

The Streamlit demo opens at `http://localhost:8501` and is designed for live evaluation. Its default offline demo backend is not used for paper results; it exists so the user interface can be demonstrated even without Ollama or API keys.

## Complete WS-DREAM Offline Evaluation

Both canonical WS-DREAM QoS releases were downloaded from the official Zenodo record and processed. Dataset #1 was evaluated with 30 pools per task. Dataset #2 was aggregated from 40,896,000 records per QoS metric and evaluated both statically and across all 64 observed time slices.

| dataset/protocol | strongest agent mean regret | lookup baseline mean regret | global baseline mean regret |
|---|---:|---:|---:|
| Dataset #1, static | 0.000190 | 0.000008 | 0.013037 |
| Dataset #2, static | 0.002682 | 0.000210 | 0.015164 |
| Dataset #2, temporal | 0.005520 | 0.000142 dynamic / 0.002640 static | 0.020500 dynamic / 0.024532 static |

The agent consistently beats the global fixed baseline, but the task-aware lookup baseline remains stronger. The temporal run confirms that automatic re-ranking helps relative to a global recommendation held static, while a task-aware lookup method re-ranked every slice remains the best condition. See `results/tables/wsdream_complete_summary.md`.

These WS-DREAM results use `DemoHeuristicBackend`, not a live LLM. They validate the complete data and agent-methodology pipeline and must be labeled as offline-controller results in any presentation.

## Honest Limitations

- No live LLM backend is currently available in this environment: `ollama` is not installed and no `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set.
- The checked results are smoke-scale, not full protocol scale.
- Complete WS-DREAM offline-controller results are present, but a reportable live-LLM WS-DREAM run is still pending.
- Regret is measured against researcher-authored reference weights, not an external ground truth.
- The positive drift result depends on a deployment-relevant re-evaluation assumption: static baselines are periodically reviewed, while the agent is invoked every round.

## External Source Checks

The project framing was checked against the public QWS Dataset page, WS-DREAM dataset descriptions, the AutoEP arXiv page, and the ICWS 2026 call/theme pages. These checks support the corrected framing that QWS v2.0 has 2,507 services, WS-DREAM releases have distinct snapshot and time-aware variants, LLM-as-algorithm-controller is an active 2025 research direction, and Services Computing is explicitly discussing agentic AI in 2026.
