from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_selection.agent import AgentController, DemoHeuristicBackend, build_backend_from_config
from agentic_selection.baselines import GLOBAL_FIXED_WEIGHTS, TASK_LOOKUP_TABLE, topsis, weighted_sum
from agentic_selection.constants import QWS_ATTRIBUTE_COLUMNS
from agentic_selection.data.preprocessing import sample_candidate_pool
from agentic_selection.drift.simulate import find_common_top_choice, simulate_drift_sequence
from agentic_selection.evaluation.metrics import adaptation_lag
from agentic_selection.tasks import TASK_PROFILES
from agentic_selection.utils import load_config


st.set_page_config(page_title="Agentic Cloud Service Selection", layout="wide")


@st.cache_data
def load_qws() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "qws_normalized.csv"
    if not path.exists():
        st.error("Processed QWS data not found. Run `python scripts/02_prepare_data.py` first.")
        st.stop()
    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_result_table(path: str) -> pd.DataFrame:
    csv_path = ROOT / path
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def build_controller(use_configured_llm: bool) -> AgentController:
    if use_configured_llm:
        cfg = load_config(ROOT / "config.yaml")
        backend = build_backend_from_config(cfg["llm"])
    else:
        backend = DemoHeuristicBackend(QWS_ATTRIBUTE_COLUMNS)
    return AgentController(
        backend=backend,
        attribute_cols=QWS_ATTRIBUTE_COLUMNS,
        memory_path=ROOT / "memory_store" / "streamlit_demo_memory.jsonl",
        tool_menu=("weighted_sum", "topsis", "skyline_then_topsis"),
        k_memory=3,
    )


def score_table(pool: pd.DataFrame, scores: pd.Series, top_n: int) -> pd.DataFrame:
    ranked = scores.sort_values(ascending=False).head(top_n)
    table = pool.loc[ranked.index, QWS_ATTRIBUTE_COLUMNS].copy()
    table.insert(0, "score", ranked)
    return table


def bar_data(weights: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {"attribute": list(weights.keys()), "weight": list(weights.values())}
    ).sort_values("weight", ascending=False)


df = load_qws()
stable_df = load_result_table("results/tables/stable_results.csv")
drift_df = load_result_table("results/tables/drift_results.csv")

st.title("Agentic Cloud Service Selection")
st.caption(
    "Interactive BTP demo: an LLM-style controller infers QoS weights and selects a deterministic MCDM ranking tool."
)

with st.sidebar:
    st.header("Scenario")
    profile_names = [p.name for p in TASK_PROFILES]
    selected_name = st.selectbox("Task profile", profile_names)
    profile = next(p for p in TASK_PROFILES if p.name == selected_name)
    custom_task = st.text_area("Task description", value=profile.description, height=150)
    pool_seed = st.number_input("Candidate pool seed", min_value=0, value=1000, step=1)
    pool_size = st.slider("Candidate pool size", min_value=5, max_value=60, value=20, step=5)
    top_n = st.slider("Show top N services", min_value=3, max_value=15, value=5)
    use_configured_llm = st.toggle(
        "Use configured live LLM backend",
        value=False,
        help="Leave off for a fast offline demo. Turn on only if Ollama/API keys are available.",
    )

pool = sample_candidate_pool(df, n=pool_size, seed=int(pool_seed))
controller = build_controller(use_configured_llm)

tabs = st.tabs(
    [
        "Viva Brief",
        "Agent Decision",
        "Baselines",
        "Drift Demo",
        "Experiment Results",
        "Memory Log",
    ]
)

with tabs[0]:
    st.subheader("Research question")
    st.write(
        "Can an agent interpret a workload description, select an appropriate "
        "QoS-ranking method, and revise its cloud-service recommendation when "
        "service quality changes?"
    )

    overview_left, overview_right = st.columns([3, 2])
    with overview_left:
        st.subheader("Decision architecture")
        architecture = pd.DataFrame(
            [
                {
                    "stage": "1. Perceive",
                    "responsibility": "Summarize the candidate pool and its QoS attributes.",
                },
                {
                    "stage": "2. Reason",
                    "responsibility": "Infer QoS weights and choose a ranking strategy from the task context.",
                },
                {
                    "stage": "3. Act",
                    "responsibility": "Run deterministic weighted-sum, TOPSIS, or skyline-then-TOPSIS ranking.",
                },
                {
                    "stage": "4. Remember",
                    "responsibility": "Store structured decisions for inspection and future retrieval.",
                },
            ]
        )
        st.dataframe(architecture, use_container_width=True, hide_index=True)
        st.caption(
            "The controller proposes weights and a strategy. Deterministic Python code performs the final ranking."
        )

    with overview_right:
        st.subheader("Validated evidence base")
        d1, d2 = st.columns(2)
        d1.metric("QWS services", "2,507")
        d2.metric("QoS attributes", "9")
        d3, d4 = st.columns(2)
        d3.metric("Test suite", "142 passed")
        d4.metric("Agent fallback", "0%")
        st.caption("QWS data was locally checked for normalized values and missing QoS fields.")

    st.subheader("What the experiments show")
    findings_left, findings_right = st.columns(2)
    with findings_left:
        stable_summary = pd.DataFrame(
            {
                "condition": ["Lookup table", "Global fixed", "Full agent", "Weights-only agent"],
                "mean_regret": [0.0069, 0.0108, 0.0350, 0.0352],
            }
        )
        st.caption("Stable QoS conditions: lower regret is better")
        st.bar_chart(stable_summary, x="condition", y="mean_regret", height=250)
    with findings_right:
        drift_summary = pd.DataFrame(
            {
                "condition": ["Full agent", "Weights-only agent", "Global fixed", "Lookup table"],
                "adaptation_lag_rounds": [0, 0, 2, 2],
            }
        )
        st.caption("QoS drift: lower adaptation lag is better")
        st.bar_chart(drift_summary, x="condition", y="adaptation_lag_rounds", height=250)

    st.subheader("Defensible conclusion")
    st.info(
        "The lookup-table baseline produced the lowest stable-setting regret in this smoke-scale evaluation. "
        "The agent switched immediately after QoS degradation, while periodically re-evaluated static baselines "
        "needed two rounds. The contribution is adaptive, auditable decision control under changing QoS, with a "
        "measured latency trade-off."
    )
    st.caption(
        "Evaluation scope: 14 tasks, 3 candidate pools per task, 4 decision conditions; "
        "6 drift profiles with one trial each. Results are smoke-scale and should be presented as such."
    )

with tabs[1]:
    decision = controller.decide(custom_task, pool)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected service", str(decision.top_service_id()))
    c2.metric("Strategy", decision.strategy)
    c3.metric("Fallback", "yes" if decision.fallback_triggered else "no")
    c4.metric("Latency", f"{decision.latency_seconds:.2f}s")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Agent weights")
        st.bar_chart(bar_data(decision.weights), x="attribute", y="weight", height=330)
    with right:
        st.subheader("Ranked services")
        st.dataframe(score_table(pool, decision.ranking, top_n), use_container_width=True)

    st.subheader("Explanation")
    st.write(decision.justification)

with tabs[2]:
    lookup_weights = TASK_LOOKUP_TABLE[profile.key]
    methods = {
        "Global fixed + TOPSIS": topsis(pool, GLOBAL_FIXED_WEIGHTS, QWS_ATTRIBUTE_COLUMNS),
        "Lookup table + TOPSIS": topsis(pool, lookup_weights, QWS_ATTRIBUTE_COLUMNS),
        "Lookup table + weighted sum": weighted_sum(pool, lookup_weights, QWS_ATTRIBUTE_COLUMNS),
        "Agent selected": decision.ranking,
    }
    rows = []
    for name, scores in methods.items():
        rows.append({"method": name, "top_service": scores.sort_values(ascending=False).index[0]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("Baseline ranking details")
    selected_method = st.selectbox("Ranking to inspect", list(methods))
    st.dataframe(score_table(pool, methods[selected_method], top_n), use_container_width=True)

with tabs[3]:
    st.write("This demo degrades the consensus top service and compares active recommendations over time.")
    rounds = st.slider("Rounds", min_value=10, max_value=30, value=20)
    degrade_start = st.slider("Degradation starts at round", min_value=2, max_value=rounds - 2, value=8)
    magnitude = st.slider("Degradation magnitude", min_value=0.1, max_value=0.95, value=0.75, step=0.05)
    reevaluation_period = st.slider("Static baseline re-evaluation period", min_value=2, max_value=10, value=5)

    consensus = find_common_top_choice(
        pool,
        {
            "global_fixed": lambda p: topsis(p, GLOBAL_FIXED_WEIGHTS, QWS_ATTRIBUTE_COLUMNS),
            "lookup_table": lambda p: topsis(p, TASK_LOOKUP_TABLE[profile.key], QWS_ATTRIBUTE_COLUMNS),
        },
    )
    if consensus is None:
        st.warning("This candidate pool has no consensus baseline top service. Try another seed.")
    else:
        seq = simulate_drift_sequence(
            pool,
            QWS_ATTRIBUTE_COLUMNS,
            target_service_id=consensus,
            degraded_attributes=profile.dominant_attributes[:2],
            n_rounds=rounds,
            degrade_start_round=degrade_start,
            magnitude=magnitude,
            profile="sudden",
            seed=int(pool_seed),
        )
        lookup_fn = lambda p: topsis(p, TASK_LOOKUP_TABLE[profile.key], QWS_ATTRIBUTE_COLUMNS).sort_values(ascending=False).index[0]
        agent_fn = lambda p: controller.decide(custom_task, p).top_service_id()
        lookup_lag = adaptation_lag(seq.pools, consensus, degrade_start, lookup_fn, reevaluation_period)
        agent_lag = adaptation_lag(seq.pools, consensus, degrade_start, agent_fn, None)

        trace_df = pd.DataFrame(
            {
                "round": list(range(rounds)),
                "lookup_active_service": [str(x) for x in lookup_lag.active_recommendation_trace],
                "agent_active_service": [str(x) for x in agent_lag.active_recommendation_trace],
                "degraded_target": str(consensus),
                "drift_started": [i >= degrade_start for i in range(rounds)],
            }
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Degraded target", str(consensus))
        c2.metric("Lookup lag", "censored" if lookup_lag.censored else f"{lookup_lag.lag_rounds} rounds")
        c3.metric("Agent lag", "censored" if agent_lag.censored else f"{agent_lag.lag_rounds} rounds")
        st.dataframe(trace_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Stable experiment")
    if stable_df.empty:
        st.info("No stable result CSV found yet.")
    else:
        agg = stable_df.groupby("condition", as_index=False)["regret"].mean().sort_values("regret")
        st.bar_chart(agg, x="condition", y="regret", height=320)
        st.dataframe(agg, use_container_width=True, hide_index=True)

    st.subheader("Drift experiment")
    if drift_df.empty:
        st.info("No drift result CSV found yet.")
    else:
        drift_clean = drift_df[~drift_df["censored"].astype(bool)].copy()
        drift_clean["lag_rounds"] = drift_clean["lag_rounds"].astype(float)
        lag = drift_clean.groupby("condition", as_index=False)["lag_rounds"].mean().sort_values("lag_rounds")
        st.bar_chart(lag, x="condition", y="lag_rounds", height=320)
        st.dataframe(lag, use_container_width=True, hide_index=True)

with tabs[5]:
    memory_path = ROOT / "memory_store" / "agent_memory.jsonl"
    if not memory_path.exists():
        st.info("No memory log found yet.")
    else:
        records = []
        for line in memory_path.read_text(encoding="utf-8").splitlines()[-50:]:
            try:
                records.append(ast.literal_eval(line))
            except Exception:
                import json

                records.append(json.loads(line))
        if records:
            compact = pd.DataFrame(
                [
                    {
                        "task": r.get("task_description", "")[:70],
                        "strategy": r.get("strategy"),
                        "fallback": r.get("fallback_triggered"),
                        "justification": r.get("justification", "")[:140],
                    }
                    for r in records
                ]
            )
            st.dataframe(compact, use_container_width=True, hide_index=True)
