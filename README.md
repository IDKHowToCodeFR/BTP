# Agentic Cloud Service Selection

Code for the paper *"Agentic Cloud Service Selection: Dynamic Strategy and
Weight Inference via LLM-Based Reasoning Agents"* (see `paper/`). An LLM
infers task-appropriate QoS attribute weights and picks a classical MCDM
strategy (weighted-sum / TOPSIS / skyline-then-TOPSIS) per request; the
ranking arithmetic itself stays deterministic and auditable. Evaluated
against static baselines on the real QWS Dataset v2.0.

## Before you run anything: read this section

This codebase was built by actually downloading and byte-inspecting the
real data sources rather than assuming their format from the paper
brief, and several real problems turned up. They're all fixed in the
code, but you should know about them because they change what the paper
can honestly claim:

1. **QWS has no monetary "cost" attribute.** Two of the six task
   profiles (Batch processing, Low-cost prototype) originally listed
   "cost" as a dominant attribute. The real dataset's 9 QoS attributes
   are response_time, availability, throughput, successability,
   reliability, compliance, best_practices, latency, documentation --
   no price field. Both profiles were redefined to use only real
   attributes (see `src/agentic_selection/constants.py` and
   `baselines/lookup_table.py` for the corrected weight vectors and
   rationale).

2. **"WS-DREAM" is not one dataset.** The brief described a single
   dataset as "5,825 services x 339 users x 64 time slices" -- that
   sentence actually conflates two different WS-DREAM releases:
   Dataset #1 (339 users x 5,825 services, single snapshot, no time
   dimension) and Dataset #2 (142 users x 4,500 services x 64 time
   slices, the actual time-aware one). See
   `src/agentic_selection/data/wsdream_loader.py` for the full
   explanation. Dataset #1 is real, verified, auto-downloadable, and is
   actively used as a second, independent real dataset for the
   supplementary validation in paper §5.4 / `scripts/07_run_wsdream_validation.py`
   (restricted to the 2 task profiles whose dominant attributes are a
   subset of Dataset #1's 2-attribute schema -- see
   `tasks/wsdream_profiles.py`). Dataset #2 has now also been downloaded
   from the official Zenodo archive and evaluated across all 64 real time
   slices by `scripts/09_run_wsdream_complete.py`. The original synthetic
   drift experiment remains separate so its controlled degradation event
   and adaptation-lag definition stay reproducible.

3. **The drift/adaptation-lag experiment as originally conceived was
   untestable.** A protocol that reruns every condition's scoring
   function fresh every round gives *every* method (including the
   "non-adaptive" static baselines) an adaptation lag of ~0, because a
   stateless scoring function reacts to new input the instant it's
   recomputed -- there's no "noticing" involved. That doesn't test what
   the paper actually claims (that static systems can't adapt *without a
   human re-running them*). The fix: static baselines are re-evaluated
   only every `static_reevaluation_period` rounds (simulating periodic
   manual review); agent conditions are re-evaluated every round
   (simulating cheap, automatic re-invocation). This produces genuinely
   different, meaningful lag numbers -- see
   `src/agentic_selection/evaluation/metrics.py`'s module docstring and
   `drift/simulate.py`.

4. **Two real parsing landmines in the QWS source file.** Exactly 2 of
   the 2,507 rows contain an extra, unescaped comma embedded inside the
   WSDL address field, which silently misaligns every column after it
   under a naive `split(",")`. `data/qws_loader.py` uses
   `split(",", 10)` instead, verified against the actual file. WS-DREAM
   Dataset #1's matrices also have a trailing-tab artifact (one extra
   all-NaN column) and use `-1` as a missing-value sentinel that must be
   converted to NaN, not treated as a real measurement -- both handled
   in `data/wsdream_loader.py`.

5. **Two more real bugs, found only by actually wiring WS-DREAM Dataset #1
   into a runnable experiment rather than stopping at "the loader has unit
   tests."** (a) `userlist.txt`/`wslist.txt` contain a handful of non-UTF-8
   bytes (a mis-encoded character in a Brazilian ministry's name, of all
   things) that crash a plain `pd.read_csv` -- fixed with
   `encoding_errors="replace"`, safe here because those text fields are
   never used computationally. (b) `pandas.Series.combine(other, min)`
   raised `"the truth value of a Series is ambiguous"` against the real
   per-service observation-count Series, despite working fine on toy
   examples -- replaced with the simpler, verified-correct
   `pd.concat([...], axis=1).min(axis=1)`. Separately, wiring in a
   dataset with only 2 attributes (instead of QWS's 9) exposed a design
   gap in the validation layer: the "is this weighting degenerate"
   threshold was a single fixed constant (0.85) calibrated for a
   9-attribute space, which would have falsely flagged completely
   ordinary 2-attribute weightings (e.g. 90/10) as hallucinated. Fixed by
   making the threshold scale with attribute count
   (`agent/validation.py::default_degenerate_threshold`). All four are
   covered by regression tests, three of them (a, b, and the QWS comma
   issue above) specifically against the real downloaded files, not just
   synthetic fixtures.

None of this is cosmetic -- items 1, 3, and 5 in particular would have
produced results that don't mean what the paper says they mean. The
fixes are implemented and unit-tested (`tests/`; current local run:
139 passed, 1 skipped, with the real downloaded QWS file verified rather
than only synthetic fixtures).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -e .
# or: pip install -r requirements.txt
```

Requires Python >= 3.9.

## Choosing an LLM backend

Edit `config.yaml`. Four options (`agent/llm_backends.py`):

| provider    | cost                        | setup |
|-------------|------------------------------|-------|
| `anthropic` | pay-per-call, cheap at Haiku tier | `export ANTHROPIC_API_KEY=...` |
| `openai`    | pay-per-call | `export OPENAI_API_KEY=...` |
| `ollama`    | free, runs locally | install [Ollama](https://ollama.com), `ollama pull llama3.1:8b` |
| `mock`      | free, not a real experiment | nothing -- returns a fixed canned response, for wiring tests only |

Start with `ollama` or `mock` to validate the whole pipeline for $0 before
switching to a paid API for your real run.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of the OODA loop (Perception, Reasoning, Controller, Memory).

## Quick start

```bash
python scripts/01_download_data.py        # real QWS dataset (+ optional WS-DREAM#1)
python scripts/02_prepare_data.py          # normalize, cache to data/processed/
python scripts/03_validate_baselines.py    # run unit tests + sanity-check the data
python scripts/04_run_stable_experiment.py --n-pools 3 --yes   # CHEAP smoke test first
python scripts/05_run_drift_experiment.py --n-trials 1 --yes   # CHEAP smoke test first
python scripts/07_run_wsdream_validation.py --n-pools 3 --yes  # CHEAP smoke test first (optional, needs WS-DREAM#1)
python scripts/06_generate_report.py
python scripts/08_generate_extended_analysis.py
python scripts/09_run_wsdream_complete.py --n-pools 30 --pool-size 20
```

If the smoke test looks sane, run the full protocol (drop `--n-pools` /
`--n-trials` to use `config.yaml`'s defaults, matching paper §4.3):

```bash
python scripts/04_run_stable_experiment.py
python scripts/05_run_drift_experiment.py
python scripts/07_run_wsdream_validation.py   # optional but recommended -- paper §5.4
python scripts/06_generate_report.py
python scripts/08_generate_extended_analysis.py
```

Both experiment scripts print an estimated LLM call count and ask for
confirmation before spending any budget, and both are **resumable** --
if a run dies partway (network blip, rate limit, laptop sleeps), just
re-run the same command; completed trials are skipped, not redone.

## Interactive BTP demo

Launch the professor-facing Streamlit demo:

```bash
python -m streamlit run app/streamlit_demo.py --server.port 8501
```

Open http://localhost:8501. The demo shows the full perceive-reason-act-
remember loop on a sampled QWS candidate pool: task profile selection,
inferred QoS weights, chosen strategy, ranked services, baseline
comparison, drift adaptation, experiment summaries, and memory-log
evidence.

By default the demo uses `DemoHeuristicBackend`, a fast offline backend
that follows the same JSON contract as the real LLM. This is for live
demonstration only, so the UI works even when Ollama/API keys are absent.
Turn on "Use configured live LLM backend" in the sidebar only when
`config.yaml` points to a working Ollama/OpenAI/Anthropic backend.

Or run everything in order:
```bash
./scripts/run_all.sh --yes
```

## Turning results into the paper

`scripts/06_generate_report.py` writes `results/tables/report_tables.md`
(Tables 5.1-5.3, plus 5.4 if `scripts/07_run_wsdream_validation.py` was
run, in Markdown) and PNG figures to `results/figures/`. Copy
the tables into `paper/agentic_cloud_selection_paper.md`, replacing the
Section 5 placeholders, then write Section 6 (Discussion) against
hypotheses H1-H4 as laid out in §4.4 -- state plainly which were
supported, partially supported, or contradicted; a null result on H1
(parity with the lookup table on stable, seen tasks) is an expected,
informative finding, not a failure, so don't soften it either way.

**The report generator refuses to run if any trial was computed on
synthetic fallback data** (i.e. the real QWS download failed at some
point) unless you pass `--allow-synthetic`, which also stamps a visible
warning into the output. This is enforced in code
(`evaluation/report.py`), not just a README promise.

`scripts/08_generate_extended_analysis.py` writes the BTP-facing analysis
pack at `results/tables/extended_analysis.md` and adds extra figures:
regret by task kind, latency by agent condition, strategy-selection
frequency from memory, and an example drift trace.

## Running the test suite

```bash
pytest tests/ -v
```

140 tests, no network or API key required for any of them. Two of
the 140 additionally exercise the real downloaded files (QWS at
`data/raw/qws/QWS_Dataset_v2.txt`, WS-DREAM Dataset #1 at
`data/raw/wsdream1/`) if present, and are skipped otherwise --
so this local checkout passes 139 tests and skips the WS-DREAM real-file
test when `data/raw/wsdream1/` is absent. A completely clean checkout
without either real dataset still passes the synthetic/unit coverage.
Covers:
every baseline algorithm against hand-computed / independently-derived
closed-form expected values (not just "runs without crashing"), the
validation/fallback state machine's every failure mode, memory
persistence and retrieval, drift simulation's three degradation
profiles, the adaptation-lag metric's two re-evaluation regimes, and
protocol resumability (re-running produces zero duplicate trials).

## Project layout

```
src/agentic_selection/
  constants.py          9 QWS attributes, cost/benefit split
  data/                 loaders (QWS, WS-DREAM), normalization, synthetic fallback
  baselines/             weighted_sum, topsis, skyline, lookup_table, wsdream_lookup_table
  agent/                 perception, LLM backends, reasoning, validation, memory, controller
  tasks/                 6 QWS task profiles + 8 held-out paraphrases (H3), wsdream_profiles (§5.4)
  drift/                 degradation injection + adaptation-lag support
  evaluation/             metrics, protocol orchestration, report generation
scripts/                 01-08, run in order (see Quick start)
app/                     Streamlit professor-facing demo
tests/                   140 tests
paper/                   the paper itself
config.yaml               LLM backend + protocol settings
```

## Known limitations (disclosed, not hidden)

- Regret is computed against researcher-authored reference weight
  vectors, not an external objective ground truth (paper §7) -- this
  was already an acknowledged limitation in the original brief and
  remains one.
- The synthetic data generator (`data/synthetic.py`) samples attributes
  independently; real QWS data has cross-attribute correlation this
  doesn't reproduce. It's a fallback for pipeline testing only, never
  used for reported results (enforced by the report generator's guard).
- WS-DREAM Dataset #2 is byte-verified against the official Zenodo archive
  and processed with a bounded-memory chunked aggregator. The complete
  offline results are in `results/tables/wsdream_complete_summary.md`.
  They use `DemoHeuristicBackend`, so they validate data handling and the
  agent methodology but are not live-LLM evidence.
- WS-DREAM Dataset #1's two available attributes (response_time,
  throughput) turn out to be almost uncorrelated in the real aggregated
  data (Pearson r ≈ 0.006) -- this caps how much regret differentiation
  `scripts/07_run_wsdream_validation.py` can show between conditions,
  regardless of method quality (see paper §3.2 and §5.4 for the full
  explanation and why the WS-DREAM-specific lookup table uses more
  aggressive 90/10 weight splits to partially compensate).
- `MODEL NOTE`: `config.yaml`'s default Anthropic model string is the
  current budget-tier Claude model as of when this was built. Model
  names change; check https://docs.claude.com if it 404s for you.
