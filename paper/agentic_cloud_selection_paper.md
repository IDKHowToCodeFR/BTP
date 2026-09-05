# Agentic Cloud Service Selection: Dynamic Strategy and Weight Inference via LLM-Based Reasoning Agents

**Status:** Implementation complete, unit-tested, and evaluated on a smoke-scale run over real QWS Dataset v2.0 data. The current repository results contain 168 stable-condition rows and 24 drift-condition rows, regenerated into `results/tables/report_tables.md` and `results/figures/` on 2026-09-04. Treat these as demonstrable BTP evaluation results, not yet as full publication-scale evidence: the full protocol in Section 4.3 requires 30 candidate-pool resamples per task and 10 drift trials per profile.

## Abstract

Cloud service selection — the task of choosing among functionally-equivalent services that differ in Quality-of-Service (QoS) attributes such as response time, availability, and reliability — has historically relied on Multi-Criteria Decision-Making (MCDM) methods such as weighted-sum scoring, TOPSIS, and skyline (Pareto) filtering. These methods require a human-specified weight vector fixed at design time and a single scoring formula applied uniformly regardless of task context or how QoS conditions evolve over time. We propose an agentic architecture in which a Large Language Model (LLM) is used not to perform the ranking arithmetic itself, but to (i) infer task-appropriate attribute weights from a natural-language task description, and (ii) select or combine classical selection strategies based on the observed shape of the current candidate pool. The system operates in a perceive-reason-act-remember loop, maintaining a persistent memory of past decisions and outcomes that informs future reasoning without requiring gradient-based retraining. We evaluate the approach against static MCDM baselines — including a task-aware lookup table, which we argue is the appropriate baseline rather than a single global weight vector — on the QWS dataset (2,507 real-world web services), across stable and drift conditions. The main adaptation-lag experiment uses a documented, seeded degradation event with a known onset. A supplementary offline-controller evaluation covers both canonical WS-DREAM QoS releases, including all 64 real time slices in Dataset #2. In the current smoke-scale QWS evaluation, the agent does not outperform the lookup table on stable or held-out task regret, but it reduces adaptation lag from 2.0 rounds for static baselines to 0.0 rounds under the stated re-evaluation regime, at a substantial latency cost. We discuss the specific conditions under which an LLM-in-the-loop controller is expected to add value over deterministic methods, and the conditions under which it is not, and report both.

**Keywords:** agentic AI, cloud service selection, large language models, multi-criteria decision-making, QoS, Services Computing

## 1. Introduction

### 1.1 Motivation

Cloud computing platforms expose large pools of functionally-equivalent services — compute instances, storage backends, API providers — that differ primarily in non-functional Quality-of-Service (QoS) attributes: response time, availability, throughput, reliability, and cost, among others. Selecting the right service for a given task is a well-studied problem in Services Computing, typically framed as a Multi-Criteria Decision-Making (MCDM) problem in which candidate services are scored and ranked according to a weighted combination of their QoS attributes.

Three limitations recur across the existing literature and industrial practice:

1. **Static weighting.** The relative importance of each QoS attribute is fixed once, by a human designer, and reused for every subsequent selection request, regardless of what that request is actually for. A weight vector tuned for a latency-sensitive streaming workload is inappropriate for a reliability-critical financial backend, yet most deployed systems apply one vector universally.
2. **Static formula choice.** A single scoring method — most commonly TOPSIS or weighted-sum — is selected at design time and never revisited, even though the relative performance of these methods is known to depend on the structure of the candidate pool (e.g., degree of conflict between attributes, number of near-dominant candidates).
3. **No adaptation to drift.** Real-world QoS measurements change over time as services degrade, scale, or experience outages. Fixed-weight, fixed-formula systems have no mechanism to notice or respond to this *without a human deciding to re-run the analysis* — critically, this is a claim about deployment practice (nobody reruns an MCDM audit on every request), not about the arithmetic itself being unable to reflect new numbers. This distinction matters and is discussed further in §3.8, because a naive evaluation protocol that reruns every method fresh on every round fails to test it at all.

Recent advances in LLM-based agentic systems — architectures in which a language model perceives environment state, reasons about which action or tool to invoke, executes that action, and retains memory across decisions — have been applied to algorithm control in adjacent domains, most notably automated hyperparameter tuning for metaheuristic optimization (Xu et al., AutoEP). We are not aware of this paradigm having been applied to cloud service selection specifically, despite the Services Computing community's own recent explicit framing of the field as entering "the era of Agentic AI" (IEEE ICWS 2026 Call for Papers).

### 1.2 Problem Statement

Given a pool of n candidate services S = {s_1, ..., s_n}, each described by a vector of m QoS attributes, and a task description T expressed in natural language, select the service (or ranked top-k list) best suited to T, under the constraint that:

- The mapping from T to attribute importance is not known in advance and must be inferred per request.
- The QoS attribute distributions of S may drift between selection requests.
- The selection decision should be accompanied by a human-readable justification.

### 1.3 Contributions

1. An agentic architecture — perceive, reason, act, remember — in which an LLM's role is precisely scoped to weight inference and strategy selection, while the ranking computation itself remains a deterministic, auditable classical algorithm (weighted-sum, TOPSIS, or skyline).
2. A memory mechanism that allows the system to improve its strategy selection across repeated decisions without retraining, based on logged outcomes.
3. An evaluation protocol that compares the agentic system not against a deliberately weak single-global-weight baseline, but against a task-aware lookup table constructed by a human — the fairer and more informative comparison — under both static and drift conditions, using two independent real-world QoS datasets, with a drift-evaluation design that is actually capable of distinguishing an adaptive from a non-adaptive method (§3.8, a correction over a naive equal-reevaluation protocol which is not).
4. An empirical smoke-run characterization of where the controller does and does not add value: drift adaptation improves under continuous agent re-evaluation, while stable-condition and held-out-task regret do not yet beat the task-aware lookup table.

### 1.4 Scope and Non-Goals

This work does not attempt to forecast future QoS values (a time-series prediction problem), does not train or fine-tune any model (the LLM is used purely at inference time via prompting), and does not claim the agent performs multi-step autonomous task execution beyond the single-decision-per-round loop described in Section 3. We use the term "agentic" in the sense established by recent algorithm-control literature (tool selection plus persistent state across decisions), and we discuss this terminological choice explicitly in Section 6.

## 2. Related Work

### 2.1 Classical Cloud/Web Service Selection

Early work on QoS-based web service selection established the foundational QWS dataset and scoring methodology (Al-Masri & Mahmoud, 2007a; 2007b), introducing a relevancy function over measured QoS attributes for ranking real-world web services. Subsequent work extended this into trust-aware selection: Pan et al. (2015) propose a trust-enhanced similarity model that predicts missing QoS values and incorporates neighbor trust into the ranking; Yang et al. (2020) apply intuitionistic fuzzy numbers to represent uncertainty in trust assessment; Li (2020) proposes FASTCloud, a framework combining interval-valued QoS attributes with an objective, deviation-maximization-based weighting scheme rather than subjective human weights — notably, still a fixed objective weighting, computed once from the dataset rather than adapted per task. Idrissi (2015) frames service selection explicitly as an agent-based process, combining skyline filtering with the ELECTRE-Is outranking method, but "agent" here refers to a rule-based software agent, not an LLM-based reasoning system, and the selection logic remains fixed regardless of task context.

Across this body of work, a consistent pattern holds: weighting schemes are either manually specified once or computed once from the static properties of the dataset, and the scoring formula is chosen once by the system designer. None of these systems reconsider either choice per request or in response to a natural-language description of task intent.

### 2.2 Multi-Agent Reinforcement Learning for Adjacent Resource-Allocation Problems

In the closely related domain of wireless rechargeable sensor networks, dynamic resource-scheduling problems (e.g., mobile charger routing and charging-ratio control) are addressed almost exclusively through Multi-Agent Reinforcement Learning (MARL), such as attention-shared multi-agent actor-critic frameworks trained to jointly optimize charging sequence and ratio. These systems adapt to dynamic conditions, but require substantial training data and retraining when the underlying distribution (e.g., network topology, node density) shifts substantially — a limitation motivating our interest in an in-context, reasoning-based alternative that does not require gradient updates to adapt.

### 2.3 LLMs as Controllers for Classical Algorithms

A distinct and more recent line of work uses LLMs not to solve an optimization problem directly, but to control a classical algorithm's configuration. AutoEP (Xu et al., 2025) uses an LLM as a zero-shot reasoning engine that reads real-time Exploratory Landscape Analysis features from a running metaheuristic and adjusts its hyperparameters accordingly, reporting that this "sense-reason-act" approach outperforms both hand-coded rules and reinforcement-learning-based hyperparameter tuners while avoiding their sample-complexity cost. The Polymorphic Metaheuristic Framework (2025) similarly uses an LLM-driven agent to switch between different metaheuristic algorithms mid-run based on real-time performance feedback. Both systems are evaluated on general-purpose combinatorial and continuous optimization benchmarks, not on any services-computing or resource-selection task.

Our work adopts the same "LLM controls classical algorithm, does not replace it" design philosophy as AutoEP and PMF, but applies it to a different problem class (multi-attribute selection among a fixed candidate pool, rather than iterative search over a solution space) and a different domain (cloud service selection), with task-context inference via natural language as an additional agent responsibility not present in either prior system.

### 2.4 Agentic AI in Services Computing

The Services Computing community has recently and explicitly signaled interest in agentic approaches: the IEEE ICWS 2026 Call for Papers frames the conference as occurring "in the era of Agentic AI" and lists dedicated tracks for agent-based service negotiation, service reasoning and intent understanding, and intent-driven orchestration. This indicates community-level interest in exactly this direction, but at the time of writing we find no published system that applies LLM-based reasoning specifically to the QoS-attribute-weighting and strategy-selection problem addressed here.

### 2.5 Positioning

Our contribution sits at the intersection of §2.1 and §2.3: we bring the "LLM as zero-shot algorithm controller" paradigm, previously demonstrated only on general optimization benchmarks, to the specific, well-established problem of QoS-based cloud service selection, and extend it with a natural-language task-interpretation responsibility and cross-decision memory not present in the metaheuristic-control literature.

## 3. Methodology

### 3.1 System Overview

The system operates as a four-stage loop executed once per selection request:

```
 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │  PERCEIVE   │ -> │   REASON    │ -> │     ACT     │ -> │  REMEMBER   │
 │  summarize  │    │  LLM infers │    │  run chosen │    │  log state, │
 │  candidate  │    │  weights +  │    │  classical  │    │  decision,  │
 │  pool state │    │  strategy   │    │  algorithm  │    │  outcome    │
 └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
        ^                                                         │
        └─────────────────────  informs next round  ──────────────┘
```

Implemented in `agentic_selection.agent`: `perception.py` (PERCEIVE), `reasoning.py` + `validation.py` (REASON), `controller.py` dispatching into `baselines/` (ACT), `memory.py` (REMEMBER).

### 3.2 Datasets

**QWS Dataset (v2.0).** 2,507 real-world web services measured on 9 QoS attributes (response time, availability, throughput, successability, reliability, compliance, best practices, latency, documentation), collected by Al-Masri and Mahmoud via the Web Service Crawler Engine. Used as the primary candidate pool for both stable-condition and drift experiments.

**WS-DREAM Dataset #1.** QoS measurements (response time and throughput only) for 5,825 web services invoked by 339 users, a single snapshot with no time dimension. Used as a second, independent real-world dataset for a supplementary validation pass (§5.4, run via `scripts/07_run_wsdream_validation.py`) restricted to the two task profiles whose dominant attributes are closest to {response_time, throughput} (Streaming, IoT telemetry ingestion), since this release only provides two of the nine QWS-style attributes.

This validation pass uses its own small, dedicated lookup table (`baselines/wsdream_lookup_table.py`) rather than mechanically renormalizing a 2-attribute slice of the 9-attribute QWS weight vectors — dropping seven attributes and rescaling the remaining two does not obviously preserve the *shape* of the original judgment. Both profiles reuse the exact same natural-language task description as their QWS counterpart (imported directly from `tasks/profiles.py`), which is deliberate: it means the LLM sees the identical sentence regardless of which attribute vocabulary underlies the pool, making this a genuine test of generalization across attribute *schemas* rather than a test of sensitivity to reworded prompts. One further finding shaped this design and is disclosed here rather than only in code comments: in the real, per-service-aggregated pool, response_time and throughput turn out to be almost perfectly uncorrelated (Pearson r ≈ 0.006, checked directly against the downloaded data). With no real trade-off tension between the two attributes, a moderately different weight split (e.g. 75/25 vs. a uniform 50/50) usually still lands on the same "good on both dimensions" top-1 candidate in a random pool — an empirical check found this made a 75/25-style split show a measurable regret difference from the uniform baseline in only a small minority of trials. The two WS-DREAM profiles therefore use a more aggressive 90/10 split (still comfortably under the validation layer's degeneracy threshold for a 2-attribute schema, §3.6) specifically to give this supplementary check enough power to be informative; readers should expect noticeably less regret differentiation here than in the main QWS results regardless, precisely because the underlying attributes don't trade off against each other the way QWS's nine do.

> **Correction relative to an earlier project brief:** "WS-DREAM" is not a single dataset. An earlier description as "5,825 services × 339 users × 64 time slices" conflates Dataset #1 (339 users × 5,825 services, one snapshot) with Dataset #2 (142 users × 4,500 services × 64 time slices). Both canonical releases were downloaded from the official Zenodo record and byte-inspected. Dataset #2 contains separate four-column response-time and throughput files with 40,896,000 rows each. The implementation aggregates them in bounded-memory chunks while preserving observation counts.

### 3.3 Data Preparation

All QoS attributes are min-max normalized to [0, 1] globally (i.e., computed once across the full dataset before any candidate pool is sampled, so a given normalized value means the same thing across every experiment run on that dataset). Attributes are labeled either benefit (higher is better, e.g., throughput, availability) or cost (lower is better, e.g., response time, latency); cost attributes are inverted after normalization (1 − x_norm) so that, post-processing, higher values are uniformly better across all attributes and algorithms. A function `sample_candidate_pool(df, n, seed)` draws a reproducible random subset of n services to instantiate a single selection request for controlled experimentation.

Two real, disclosed data-quality issues were found and handled explicitly while implementing this pipeline (both would silently corrupt results if missed, and are covered by regression tests): (a) exactly 2 of the QWS dataset's 2,507 rows contain an extra, unescaped comma embedded inside the WSDL address field, which misaligns every subsequent column under naive comma-splitting — fixed by capping the split at the expected field count; (b) WS-DREAM Dataset #1's response-time and throughput matrices use `-1` as a missing-value sentinel (~5.1% and ~7.3% of cells respectively) and carry a trailing-tab artifact producing one spurious all-NaN column, both handled before any statistics are computed on the matrices. Two further bugs surfaced only once WS-DREAM Dataset #1 was actually wired into a runnable experiment rather than stopping at an isolated loader: (c) `userlist.txt`/`wslist.txt` contain a handful of non-UTF-8 bytes in provider/location name fields, which crash a plain UTF-8 CSV read — fixed with error-tolerant decoding, safe because those text fields are never used computationally; (d) computing the minimum valid-observation count across the two matrices via `pandas.Series.combine(other, min)` raised an unrelated-looking error against the real data's actual Series shape despite working on toy examples, and was replaced with a simpler, verified-correct `concat` + `min(axis=1)`. All four are regression-tested against the real downloaded files, not only small synthetic fixtures.

### 3.4 Baseline Selection Methods

Three deterministic algorithms are implemented as callable, parameterized functions (`agentic_selection.baselines`) — these are never modified by the LLM; only their inputs (weights) or which one is invoked (strategy choice) is under agent control.

**Weighted Sum.** For candidate i with normalized attributes x_i,1, ..., x_i,m and weight vector w (with Σw_j = 1):

```
score(i) = Σ_j w_j · x_i,j
```

**TOPSIS.** Computes each candidate's Euclidean distance to an ideal point A+ (the per-attribute max of the *weighted* matrix) and a worst-case point A− (the per-attribute min), ranking candidates by:

```
closeness(i) = d(i, A−) / (d(i, A+) + d(i, A−))
```

Because attributes are already normalized and cost-inverted (§3.3) before reaching this function, there is no remaining cost/benefit branching to do inside TOPSIS itself — a standard, disclosed simplification sometimes called "TOPSIS on a pre-normalized matrix," validated against a hand-computed worked example (`tests/test_topsis.py`, exact to 1e-9) rather than assumed correct.

**Skyline.** Returns the subset of candidates not dominated on every attribute simultaneously by any other candidate; used either standalone (returning a non-dominated set for the user to review) or as a pre-filter before applying TOPSIS to the reduced candidate set (`skyline_then_topsis`).

Each is validated against a hand-computed worked example before being used in any experiment, to confirm correctness independent of any agent behavior (`tests/test_weighted_sum.py`, `tests/test_topsis.py`, `tests/test_skyline.py`).

### 3.5 Baselines for Comparison

Two baseline conditions are used, in increasing order of strength:

1. **Global fixed weight** — a single weight vector, chosen once, applied to all task types. To avoid a strawman comparison (reviewers in this space know MCDM well and will spot one immediately), this is implemented as a uniform vector (1/9 on every QWS attribute) — the least informative honest choice a designer with no task-awareness at all could make, not a deliberately bad one.
2. **Task-aware lookup table** — a set of hand-authored weight vectors, one per predefined task profile (§3.7), chosen by the researcher with the same domain reasoning the LLM is asked to perform, but fixed and non-adaptive to drift or to task descriptions outside the predefined set. This is the primary, fair comparison point.

### 3.6 Agent Design

**Perception module.** Given a candidate pool, computes a compact numerical summary: pool size, per-attribute variance, an inter-attribute conflict score (mean magnitude of negative pairwise correlations — strongly negative correlation between two attributes indicates a genuine trade-off), an outlier fraction (rows with ≥1 attribute outside 1.5×IQR), and the fraction of missing values.

**Reasoning module.** A prompt template supplies the LLM with (a) the natural-language task description, (b) the perception summary, (c) a fixed menu of available tools (weighted-sum, TOPSIS, skyline-then-TOPSIS), and (d) a condensed digest of the two or three most similar past decisions and their logged outcomes (from the memory store, via nearest-neighbor retrieval over the perception vector). The model is instructed to return a structured JSON object:

```json
{
  "weights": {"response_time": 0.15, "availability": 0.35, ...},
  "strategy": "topsis",
  "justification": "Task specifies transactional reliability as critical; ..."
}
```

A deterministic validation step (`agent/validation.py`) checks that: the response is parseable JSON; `strategy` is one of the three allowed tool names; every key in `weights` is a real attribute of the current pool (an unrecognized key — e.g. a hallucinated attribute — is treated as a failure, not silently dropped); no weight is negative; the weights do not sum to (numerically) zero; and, after renormalizing to sum to 1, no single attribute holds more than a threshold fraction of the total mass. That threshold scales with the number of attributes in play rather than being one fixed constant: 85% for QWS's 9-attribute schema (deliberately generous — the most concentrated hand-authored profile in this project's own lookup table, §3.7, tops out around 32–34% on any single attribute, so 85% is not a plausible value for a genuinely task-appropriate weighting there, only for a degenerate or hallucinated one) rising to 95% for a 2-attribute schema such as WS-DREAM Dataset #1's (§5.4), where a lopsided 90/10 split is completely ordinary rather than suspicious, and a fixed 85% cutoff was found, while integrating that dataset, to falsely flag such legitimate weightings as degenerate (`agent/validation.py::default_degenerate_threshold`). Any validation failure reverts to the task-aware lookup table's weights and a default strategy (TOPSIS); this fallback's trigger rate is itself logged as an evaluation metric (§4.1). Weights that merely fail to sum to exactly 1.0 are *not* a failure and are simply renormalized.

**Action module.** Invokes the selected deterministic function with the validated weights and returns a ranked list.

**Memory module.** After each decision, the tuple (task description, perception vector, decision, later-observed outcome) is appended to a persistent JSON Lines log. Retrieval for future prompts uses nearest-neighbor similarity (Euclidean distance) over the perception vector; no vector database is used given the small scale of this study.

### 3.7 Task Profiles

Six natural-language task profiles are defined, each paired with a researcher-authored reference weight vector used both to construct the lookup-table baseline and as an approximate (not objective) ground truth for evaluation:

| Profile | Description | Dominant attributes |
|---|---|---|
| Streaming | Real-time video/audio delivery | Throughput, response time, latency |
| Financial transaction | Banking/payments backend | Reliability, availability |
| Batch processing | Large offline data jobs | Throughput, successability |
| Low-cost prototype | Early-stage/hobby hosting | Documentation, best practices |
| IoT telemetry ingestion | High-frequency small messages | Response time, successability |
| Compliance-sensitive backend | Regulated industries | Compliance, best practices |

> **Correction relative to an earlier project brief:** the "Batch processing" and "Low-cost prototype" rows originally listed a "cost" attribute as dominant. QWS has no monetary/price QoS attribute — fabricating one would mean scoring against a synthetic field dressed up as real QoS data — so both were redefined using only attributes that genuinely exist in QWS. Batch processing now weights throughput and successability (a large unattended job needs to both move volume and actually finish without failing partway through, unwatched); Low-cost prototype now weights documentation and best-practice conformance (a hobbyist without a dedicated ops team leans on self-service integration quality rather than raw performance guarantees). Full weight vectors are in `baselines/lookup_table.py`, and a regression test (`test_no_cost_attribute_referenced_anywhere`) enforces that no lookup-table entry ever references a non-existent attribute again.

A held-out set of eight rephrased and novel task descriptions (paraphrases in a different domain, deliberate blends of two profiles' concerns, and one scenario with no close analogue among the six profiles — see `tasks/profiles.py::HELD_OUT_TASKS`) is used specifically to test generalization beyond the lookup table's fixed coverage (§4.2, hypothesis H3).

### 3.8 Drift Simulation

**Why the controlled drift protocol remains necessary.** Dataset #2 is now available and evaluated separately, but its 64 slices do not label a specific service-degradation event or onset. A protocol that reruns every condition's scoring function fresh on every round also gives every method immediate access to the new numbers. The main adaptation-lag experiment therefore retains a controlled degradation event with a known start round, while the Dataset #2 extension reports regret, top-1 accuracy, and recommendation switches over real temporal observations.

**The corrected design.** Controlled drift scenarios are constructed on real, verified QWS data rather than an external, unverified time-series file: a candidate pool is sampled, and a service that every static baseline currently agrees is the #1 choice is identified (`drift.find_common_top_choice`; if no such consensus candidate exists for a given random pool, a different seed is tried — not every pool has one, which is expected). One or more of that service's dominant attributes are then degraded over a sequence of synthetic rounds according to an explicit, disclosed curve — `sudden` (an immediate drop at a chosen round), `gradual` (a linear ramp), or `sudden_then_recover` (a drop followed by a return to the original value, testing whether a method whipsaws back prematurely or correctly waits out a real degradation) — while every other cell in the pool is held fixed.

Static baselines (global fixed, lookup table) are then re-evaluated only every `K` rounds, simulating a person periodically re-running the MCDM analysis; agent conditions are re-evaluated every round, simulating automatic, cheap re-invocation on each request. **Adaptation lag** is the number of rounds after degradation onset until each condition's *active* recommendation (not a freshly-recomputed-but-never-actually-deployed score, but what the condition would actually be telling a user to use right now, given its own re-evaluation cadence) stops being the degraded service. This asymmetry — continuous re-evaluation for the agent, periodic for the static baselines — is not an artificial handicap imposed on the baselines; it is the literal mechanism by which an agentic system could out-adapt a static one in practice (an LLM call is cheap enough to make on every request; a human manually re-running an audit is not), and it is precisely why decision cost (H4) is reported alongside this metric rather than as an unrelated number: the trade-off the paper wants to surface is faster adaptation *at a non-trivial marginal cost per decision*, not a free win.

The complete secondary runner (`scripts/09_run_wsdream_complete.py`) consumes genuine Dataset #2 measurements and compares held-static baselines, baselines re-ranked every slice, and agent conditions re-evaluated every slice.

### 3.9 Reproducibility and Implementation

The full system described above is implemented (not just specified) as a tested Python package. 140+ unit and integration tests cover: every baseline algorithm against hand-computed or independently-derived closed-form expected values (not merely "runs without an exception"); every failure mode of the validation/fallback state machine (malformed JSON, unknown attribute names, negative weights, zero-sum weights, degenerate concentration at both the 9-attribute and 2-attribute thresholds); memory persistence and nearest-neighbor retrieval; all three drift degradation profiles; both re-evaluation regimes of the adaptation-lag metric; protocol resumability (re-running an interrupted experiment produces zero duplicate trials); and the offline demo backend used by the Streamlit showcase. Tests additionally exercise the real, downloaded data end-to-end when those files are present. Both the stable-condition and drift experiment scripts print an estimated LLM call count and require confirmation before spending API budget, and both write results incrementally so an interrupted run can be resumed rather than restarted. The repository also includes `app/streamlit_demo.py`, an interactive BTP demonstration showing task selection, inferred weights, chosen strategy, ranked services, baseline comparison, drift adaptation, experiment summaries, and memory-log evidence. See the accompanying code repository's `README.md` for setup and exact commands.

## 4. Experimental Design

### 4.1 Evaluation Metrics

- **Regret.** For a given task profile and candidate pool, the difference between the score (under the task's reference weight vector, computed as a simple weighted sum — a fixed, method-independent yardstick) of the reference-optimal candidate and the score of the top choice actually returned by each method.
- **Adaptation lag.** Number of rounds after a simulated drift event until a method's *active* recommendation stops being the now-degraded service, under the re-evaluation regime described in §3.8 (continuous for agent conditions, periodic for static baselines). Trials in which a condition never stops recommending the degraded service within the observation window are right-censored and reported as a separate count, not silently dropped or averaged in as if adaptation had occurred.
- **Stability.** Variance in the top-ranked candidate across repeated identical runs (applicable to the agent only, since the LLM's output is not fully deterministic even at low temperature). Operationalized as 1 − (fraction of runs agreeing with the modal choice).
- **Fallback trigger rate.** Fraction of agent decisions that fail validation and revert to the lookup-table fallback (§3.6) — a direct measure of how often the LLM component fails outright.
- **Decision cost.** Wall-clock latency and number of LLM API calls per decision, reported alongside all quality metrics.
- **Justification quality (qualitative).** A sample of ~20 agent justifications rated by the author (and, where possible, a second independent rater) for whether the stated reasoning is consistent with the weights actually returned.

### 4.2 Conditions Compared

| Condition | Weights source | Strategy source | Adapts to drift? | Handles unseen task phrasing? |
|---|---|---|---|---|
| Global fixed weight | Fixed, once | Fixed (TOPSIS) | No | No |
| Task-aware lookup table | Hand-authored, per profile | Fixed (TOPSIS) | No | No |
| Agent (weights only) | LLM-inferred | Fixed (TOPSIS) | Partial | Yes |
| Agent (full) | LLM-inferred | LLM-selected | Yes | Yes |

The two agent rows form an ablation isolating the marginal contribution of strategy selection over weight inference alone (both implemented via `AgentController.decide(..., strategy_override="topsis")` for the weights-only row).

### 4.3 Protocol

For each of the six defined task profiles and the eight held-out unseen-phrasing tasks, 30 independent candidate pools of size n = 20 are sampled from QWS. Each of the four conditions in Table 4.2 is run on every pool; regret is computed and aggregated with 95% confidence intervals (`evaluation/protocol.py::run_stable_protocol`). The drift experiment (`run_drift_protocol`) is run separately: for each of the six task profiles, 10 independent trials each sample a fresh pool, locate a consensus degradation target, run a 20-round sequence with degradation starting at round 8, and report adaptation lag per condition under the re-evaluation regime described in §3.8. Both protocols are resumable and write results incrementally to CSV.

### 4.4 Hypotheses

Stated in advance of running any experiment:

- **H1 (stable, seen tasks):** The full agent condition will show regret statistically indistinguishable from the task-aware lookup table — i.e., no significant advantage under conditions the lookup table was designed to handle.
- **H2 (drift):** The full agent condition will show meaningfully lower adaptation lag than both static baselines, which by construction cannot adapt at all without a human re-running the analysis (measured under the re-evaluation-regime design in §3.8, not the trivial "everyone recomputes every round" design that cannot actually distinguish the conditions).
- **H3 (unseen task phrasing):** The full agent condition will show lower regret than the lookup table on the held-out paraphrase set, where the lookup table has no matching entry and must fall back to a fuzzy string match or a default heuristic (`baselines/lookup_table.py::get_lookup_weights`, fallback="nearest").
- **H4 (cost):** The agent will incur non-trivial per-decision latency and API cost relative to the near-zero cost of the deterministic baselines, and this cost should be explicitly weighed against the gains in H2/H3 rather than treated as negligible.

## 5. Results

The results below are generated from the repository's current smoke-scale experiment artifacts: `results/tables/stable_results.csv` and `results/tables/drift_results.csv`, regenerated into `results/tables/report_tables.md` on 2026-09-04. The run uses real QWS data (`is_synthetic_data=False`) but only three stable candidate-pool resamples per task and one drift trial per canonical profile. Therefore, the tables are suitable for project demonstration and methodological evaluation; publication-scale claims require the full protocol counts in Section 4.3.

### 5.1 Regret Across Task Profiles (Stable Conditions)

Table 5.1 reports regret against the task-reference optimum. Lower is better.

| task                         | Global fixed                   | Lookup table                   | Agent (weights only)           | Agent (full)                   |
|:-----------------------------|:-------------------------------|:-------------------------------|:-------------------------------|:-------------------------------|
| batch_processing             | 0.0099 [-0.0327, 0.0525] (n=3) | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0305 [-0.0462, 0.1072] (n=3) | 0.0305 [-0.0462, 0.1072] (n=3) |
| compliance_sensitive_backend | 0.0247 [-0.0817, 0.1312] (n=3) | 0.0043 [-0.0143, 0.0229] (n=3) | 0.0077 [-0.0135, 0.0290] (n=3) | 0.0021 [-0.0069, 0.0112] (n=3) |
| financial_transaction        | 0.0144 [-0.0476, 0.0764] (n=3) | 0.0025 [-0.0084, 0.0134] (n=3) | 0.0058 [-0.0069, 0.0185] (n=3) | 0.0058 [-0.0069, 0.0185] (n=3) |
| held_out_0                   | 0.0112 [-0.0369, 0.0593] (n=3) | 0.0112 [-0.0369, 0.0593] (n=3) | 0.0600 [-0.0792, 0.1991] (n=3) | 0.0600 [-0.0792, 0.1991] (n=3) |
| held_out_1                   | 0.0144 [-0.0476, 0.0764] (n=3) | 0.0144 [-0.0476, 0.0764] (n=3) | 0.0177 [-0.0386, 0.0740] (n=3) | 0.0033 [-0.0107, 0.0173] (n=3) |
| held_out_2                   | 0.0099 [-0.0327, 0.0525] (n=3) | 0.0099 [-0.0327, 0.0525] (n=3) | 0.0305 [-0.0462, 0.1072] (n=3) | 0.0099 [-0.0327, 0.0525] (n=3) |
| held_out_3                   | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0504 [-0.1244, 0.2252] (n=3) | 0.0878 [-0.1011, 0.2768] (n=3) |
| held_out_4                   | 0.0085 [-0.0120, 0.0289] (n=3) | 0.0085 [-0.0120, 0.0289] (n=3) | 0.0143 [-0.0295, 0.0581] (n=3) | 0.0143 [-0.0295, 0.0581] (n=3) |
| held_out_5                   | 0.0247 [-0.0817, 0.1312] (n=3) | 0.0247 [-0.0817, 0.1312] (n=3) | 0.0709 [-0.1012, 0.2429] (n=3) | 0.0709 [-0.1012, 0.2429] (n=3) |
| held_out_6                   | 0.0144 [-0.0476, 0.0764] (n=3) | 0.0144 [-0.0476, 0.0764] (n=3) | 0.0106 [-0.0350, 0.0562] (n=3) | 0.0106 [-0.0350, 0.0562] (n=3) |
| held_out_7                   | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0000 [0.0000, 0.0000] (n=3)  | 0.1240 [-0.1431, 0.3912] (n=3) | 0.1240 [-0.1431, 0.3912] (n=3) |
| iot_telemetry_ingestion      | 0.0085 [-0.0120, 0.0289] (n=3) | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0203 [-0.0015, 0.0420] (n=3) | 0.0203 [-0.0015, 0.0420] (n=3) |
| low_cost_prototype           | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0068 [-0.0224, 0.0360] (n=3) | 0.0504 [-0.1244, 0.2252] (n=3) | 0.0504 [-0.1244, 0.2252] (n=3) |
| streaming                    | 0.0112 [-0.0369, 0.0593] (n=3) | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0000 [0.0000, 0.0000] (n=3)  | 0.0000 [0.0000, 0.0000] (n=3)  |

Aggregated across all 42 task-pool combinations per condition, the lookup table has the lowest mean regret (0.0069), followed by the global fixed baseline (0.0108), the full agent (0.0350), and the weights-only agent (0.0352). On canonical seen profiles only, the lookup table remains best (0.0023 mean regret), while the full agent averages 0.0182. On held-out tasks, the lookup table and global fixed baseline both average 0.0104, while the full agent averages 0.0476. In this smoke run, therefore, H1 is not supported as parity, and H3 is contradicted.

### 5.2 Adaptation Lag Under Drift

Table 5.2 reports rounds elapsed after degradation onset before the active recommendation stops being the degraded service.

| condition            | mean_adaptation_lag           |   n_trials |   n_censored | censored_note   |
|:---------------------|:------------------------------|-----------:|-------------:|:----------------|
| Global fixed         | 2.0000 [2.0000, 2.0000] (n=6) |          6 |            0 |                 |
| Lookup table         | 2.0000 [2.0000, 2.0000] (n=6) |          6 |            0 |                 |
| Agent (weights only) | 0.0000 [0.0000, 0.0000] (n=6) |          6 |            0 |                 |
| Agent (full)         | 0.0000 [0.0000, 0.0000] (n=6) |          6 |            0 |                 |

The drift result supports H2 under the paper's deployment-relevant re-evaluation regime: agent conditions are re-evaluated every round, while static baselines are re-evaluated every five rounds to represent periodic manual review. Both agent variants stop recommending the degraded service immediately after the degradation event; both static baselines lag by two rounds in all six observed trials.

### 5.3 Fallback Trigger Rate and Cost

Table 5.3: Operational metrics for the agent conditions.

| condition            |   fallback_trigger_rate |   mean_latency_seconds |   mean_api_calls |   n_decisions |
|:---------------------|------------------------:|-----------------------:|-----------------:|--------------:|
| Agent (weights only) |                       0 |                70.7622 |                1 |            42 |
| Agent (full)         |                       0 |                40.7552 |                1 |            42 |

The validation/fallback layer did not trigger on any stable-run agent decision. H4 is supported: the deterministic baselines run at effectively zero marginal decision cost, while the agent incurs one model call per decision and tens of seconds of latency in the recorded run.

### 5.4 Supplementary Validation on WS-DREAM Dataset #1

The complete offline-controller run uses 30 candidate pools per profile. On Dataset #1, mean regret is 0.000008 for the task-aware lookup table, 0.000190 for the weights-only agent, 0.001210 for the full agent, and 0.013037 for the global fixed baseline. On the time-aggregated Dataset #2 pool, the corresponding values are 0.000210, 0.002682, 0.002729, and 0.015164. The agent therefore generalizes substantially better than a uniform global weighting, but does not beat the task-aware lookup table.

Across all 64 real Dataset #2 time slices, the dynamically re-ranked lookup baseline achieves the lowest mean regret (0.000142). The full agent records 0.005520 and the weights-only agent 0.006590, compared with 0.020500 for the dynamically re-ranked global baseline and 0.024532 for the held-static global baseline. The held-static task-aware lookup baseline records 0.002640. These results distinguish deployment cadence from ranking quality: automatic agent re-evaluation beats the generic global baselines, while a task-aware deterministic method re-ranked every slice remains strongest.

These supplementary results use `DemoHeuristicBackend`, a deterministic offline controller that follows the same structured agent contract. They validate the complete WS-DREAM ingestion, reasoning, ranking, and temporal-evaluation pipeline, but they are not evidence from a live LLM. The reportable live-LLM WS-DREAM run remains pending until a backend is configured. Full tables are in `results/tables/wsdream_complete_summary.md`.

### 5.5 Stability

The current CSV outputs do not include a separate repeated-identical-prompt stability experiment. However, the stable smoke run uses temperature 0.0 and records zero validation fallbacks. Stability should still be evaluated explicitly before publication by repeatedly invoking `AgentController.decide(...)` on the same task-pool pair and reporting variance in the top-ranked service.

### 5.6 Qualitative Justification Review

A manual inspection of the memory log shows that generated justifications generally match the returned weights on obvious profiles. For example, streaming decisions assign high mass to throughput and latency and justify that choice by reference to sustained delivery and low end-to-end delay. A formal rated justification study has not yet been run and should not be claimed from the current artifacts.

## 6. Discussion

The smoke-run evidence narrows the claim usefully. The agent is implemented and behaves coherently, but it does not currently show a stable-regret advantage over the task-aware lookup table. This is important because the lookup table is intentionally a strong baseline: it encodes human knowledge for the six canonical profiles and uses a nearest-profile fallback for held-out descriptions. In the current run, the lookup table is not a strawman; it is the method to beat, and the agent has not beaten it on H1 or H3.

The clearest positive result is H2. When QoS degradation occurs, the agent conditions adapt immediately because they are invoked on every round, while the static methods retain a stale active recommendation until their next periodic re-evaluation. This result should be framed as a deployment-cadence advantage, not as proof that TOPSIS or weighted-sum arithmetic cannot react to new data. If all methods were recomputed every round, the adaptation-lag distinction would largely disappear; the meaningful comparison is between automatic per-request re-evaluation and periodic manual review.

Strategy selection adds little in this run. The full agent and weights-only ablation have nearly identical aggregate regret (0.0350 vs. 0.0352) and identical drift lag (0.0), so the present evidence attributes the adaptation gain mainly to continuous re-invocation rather than to choosing among weighted-sum, TOPSIS, and skyline-then-TOPSIS. This does not invalidate the architecture, but it does weaken any claim that LLM-based formula selection is already empirically important.

The reliability story is mixed but honest. A zero fallback-trigger rate suggests the schema validation and prompt contract were sufficient for the recorded stable run. The cost story is less favorable: mean latency is tens of seconds per decision, compared with effectively zero for deterministic baselines. For high-frequency service selection, this cost is hard to justify unless drift sensitivity or natural-language flexibility is operationally valuable.

## 7. Limitations

- **No objective ground truth.** The reference weight vectors used to compute regret are researcher-authored, not derived from an external, verifiable notion of "correct" task priorities. Results should be read as relative comparisons under a stated, disclosed set of assumptions, not as absolute measures of decision quality.
- **Single LLM, single sampling temperature.** Results may not generalize across different underlying models or decoding settings without separate validation.
- **Limited task profile coverage.** Six profiles plus eight held-out paraphrases is a modest test of generalization; a larger and more diverse task-description corpus would strengthen the generalization claims in H3.
- **Controlled versus observed drift.** The primary adaptation-lag experiment uses a documented, seeded degradation event because lag requires a known onset and target. The supplementary Dataset #2 evaluation uses all 64 real temporal slices but reports regret, top-1 accuracy, and recommendation switches rather than assigning an unsupported degradation-onset label.
- **WS-DREAM Dataset #1's reduced attribute schema and low inter-attribute conflict.** Only two of the nine QWS-style attributes (response time, throughput) are available from this dataset, limiting supplementary validation (§5.4) to the subset of task profiles whose dominant attributes fall within that pair. Compounding this, the two available attributes were found to be almost uncorrelated in the aggregated data (§3.2) — meaning there is little genuine trade-off tension for a weighting scheme to matter on, which structurally caps how much regret differentiation this supplementary check can show between conditions regardless of method quality. This limits its power as a discriminating test, though it remains informative as a check that the agent does not behave erratically or produce degenerate/hallucinated weights on a completely different real dataset.
- **Single-agent, single-decision-per-round scope.** As discussed in Section 1.4, this system does not perform multi-step autonomous task execution; readers expecting a more expansive sense of "agentic AI" should calibrate expectations accordingly.

## 8. Conclusion and Future Work

We have presented an agentic architecture for cloud service selection in which an LLM is scoped specifically to two decisions — attribute weighting and strategy selection — while all ranking computation remains deterministic and auditable. This design is directly motivated by, and extends to a new domain, the "LLM controls classical algorithm" paradigm previously demonstrated in general-purpose metaheuristic optimization. Building the full system surfaced and required resolving several concrete gaps between the original problem framing and the real data/evaluation mechanics — a nonexistent monetary attribute in the QWS schema, a conflation of two differently-shaped WS-DREAM releases, and a drift-evaluation design that, in its naive form, could not actually distinguish an adaptive system from a static one — each documented and corrected in place (§3.2, §3.7, §3.8) rather than glossed over, which we consider part of the contribution. The current QWS evaluation gives a bounded conclusion: the agent's strongest observed advantage is immediate adaptation under controlled drift, while regret under stable and held-out task conditions remains worse than the hand-authored lookup table. The complete offline WS-DREAM extension reaches the same broader conclusion across both static and genuine temporal data: the agent beats generic global baselines but not a task-aware lookup method. Future work includes extending memory to trigger autonomous re-selection following a detected SLA violation, testing multiple underlying LLMs, and rerunning the WS-DREAM protocol with a reachable live backend.

## References

1. Al-Masri, E., and Mahmoud, Q. H. "QoS-based Discovery and Ranking of Web Services." *IEEE 16th International Conference on Computer Communications and Networks (ICCCN)*, 2007, pp. 529–534.
2. Al-Masri, E., and Mahmoud, Q. H. "Discovering the best web service." *ACM 16th International Conference on World Wide Web (WWW)*, 2007, pp. 1257–1258.
3. Pan, Y., Ding, S., Fan, W., Li, J., and Yang, S. "Trust-Enhanced Cloud Service Selection Model Based on QoS Analysis." *PLOS ONE*, 2015. DOI: 10.1371/journal.pone.0143448.
4. Li, X. "FASTCloud: A framework of assessment and selection for trustworthy cloud service based on QoS." arXiv preprint arXiv:2011.01871, 2020.
5. Yang, Y., Yu, N., and Chen, Y. "Trusted Cloud Service Selection Algorithm Based on Lightweight Intuitionistic Fuzzy Numbers." *IEEE Access*, vol. 8, 2020, pp. 97748–97756.
6. Idrissi, A. "An Agent-based method for Cloud service selection combining Skyline and ELECTRE-Is." *International Journal of Advanced Computer Science and Applications (IJACSA)*, vol. 6, no. 6, 2015. arXiv:1702.04966.
7. Xu, Z., et al. "AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms." arXiv preprint arXiv:2509.23189, 2025.
8. "RAG/LLM Augmented Switching Driven Polymorphic Metaheuristic Framework." arXiv preprint arXiv:2505.13808, 2025.
9. Zheng, Z., Zhang, Y., and Lyu, M. R. "WS-DREAM: A QoS Dataset for Web Service Research." Dataset #1 (339 users × 5,825 services) and Dataset #2 (142 users × 4,500 services × 64 time slices) are distinct releases — see §3.2. Project page: wsdream.github.io.
10. IEEE World Congress on SERVICES. "ICWS 2026 Call for Papers." services.conferences.computer.org/2026/icws/icws-call-for-papers/.

---
*This draft includes smoke-scale live-agent QWS results and a complete offline-controller WS-DREAM evaluation. A live-LLM WS-DREAM rerun and formal stability/justification study remain necessary before publication.*
