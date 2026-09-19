#!/usr/bin/env python3
"""Run the WS-DREAM Dataset #1 supplementary validation (paper section 5.4): the
same 4-condition comparison as scripts/04, but against a completely
different, independently-sourced real dataset with a reduced 2-attribute
schema (response_time, throughput only -- see data/wsdream_loader.py),
restricted to the 2 task profiles that have a natural analogue in that
schema (streaming, iot_telemetry_ingestion -- see tasks/wsdream_profiles.py).
This is a check that the agent's behavior on QWS isn't an artifact of
QWS's specific 9-attribute schema.

Requires WS-DREAM Dataset #1 to already be downloaded:
    python scripts/01_download_data.py            # (don't pass --skip-wsdream)
    python scripts/02_prepare_data.py

Usage:
    python scripts/07_run_wsdream_validation.py
    python scripts/07_run_wsdream_validation.py --n-pools 3 --yes   # cheap smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_selection.agent import (
    AgentController,
    DemoHeuristicBackend,
    build_backend_from_config,
)
from agentic_selection.baselines.wsdream_lookup_table import (
    WSDREAM_ATTRIBUTE_COLUMNS,
    WSDREAM_GLOBAL_FIXED_WEIGHTS,
    WSDREAM_TASK_LOOKUP_TABLE,
    get_wsdream_lookup_weights,
)
from agentic_selection.data.preprocessing import normalize_benefit_oriented
from agentic_selection.data.wsdream_loader import (
    load_wsdream_dataset1,
    wsdream1_to_candidate_pool,
)
from agentic_selection.evaluation.protocol import run_stable_protocol, STABLE_RESULT_FIELDS
from agentic_selection.evaluation.storage import CsvStorage
from agentic_selection.tasks.wsdream_profiles import WSDREAM_TASK_PROFILES
from agentic_selection.utils import load_config, setup_logging


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-pools", type=positive_int, default=None)
    parser.add_argument("--pool-size", type=positive_int, default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--demo-backend",
        action="store_true",
        help=(
            "Use the deterministic offline demo controller. This verifies the "
            "WS-DREAM pipeline but must not be reported as a live-LLM result."
        ),
    )
    args = parser.parse_args()

    setup_logging()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)
    data_dir = project_root / config["paths"]["data_dir"]

    wsdream_dir = data_dir / "raw" / "wsdream1"
    required = ("userlist.txt", "wslist.txt", "rtMatrix.txt", "tpMatrix.txt")
    if not all((wsdream_dir / f).exists() for f in required):
        print(
            f"WS-DREAM Dataset #1 not found in {wsdream_dir}. Run:\n"
            f"  python scripts/01_download_data.py\n"
            f"(without --skip-wsdream) first.",
            file=sys.stderr,
        )
        return 1

    wcfg = config["protocol"]["wsdream"]
    n_pools = args.n_pools if args.n_pools is not None else wcfg["n_pools"]
    pool_size = args.pool_size if args.pool_size is not None else wcfg["pool_size"]

    print(f"Loading WS-DREAM Dataset #1 from {wsdream_dir} ...")
    wsdream1 = load_wsdream_dataset1(wsdream_dir)
    raw_pool = wsdream1_to_candidate_pool(
        wsdream1,
        min_observations=wcfg["min_observations"],
    )
    print(
        f"  {len(raw_pool)} services with >= {wcfg['min_observations']} "
        "valid user observations"
    )

    norm_pool = normalize_benefit_oriented(
        raw_pool, attribute_cols=WSDREAM_ATTRIBUTE_COLUMNS, cost_attributes={"response_time"}
    )

    n_tasks = len(WSDREAM_TASK_PROFILES)
    est_calls = n_tasks * n_pools * 2
    provider = "demo_heuristic" if args.demo_backend else config["llm"]["provider"]

    print("\n== WS-DREAM supplementary validation plan ==")
    print(f"  task profiles (subset with a 2-attribute analogue): {n_tasks}")
    print(f"  pools per task                 : {n_pools} (size {pool_size})")
    model = config["llm"].get("model") if not args.demo_backend else "not used"
    print(f"  LLM provider                   : {provider} (model: {model})")
    print(
        f"  estimated LLM calls            : ~{est_calls} "
        f"({n_tasks} tasks x {n_pools} pools x 2 agent conditions)"
    )
    if provider not in ("mock", "demo_heuristic") and est_calls > 0 and not args.yes:
        resp = input(f"\nProceed with ~{est_calls} real LLM calls via '{provider}'? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    if args.demo_backend:
        print(
            "NOTE: using the offline demo controller. Output validates the "
            "cross-dataset pipeline only and is not live-LLM evidence."
        )
        backend = DemoHeuristicBackend(WSDREAM_ATTRIBUTE_COLUMNS)
    else:
        backend = build_backend_from_config(config["llm"])
    # Deliberately a SEPARATE memory file from the QWS run: WS-DREAM's
    # perception vectors have a different dimensionality (2-attribute
    # schema vs QWS's 9), and while memory.nearest() already skips
    # mismatched-shape records safely, keeping them in fully separate
    # files avoids any ambiguity about which experiment a logged decision
    # belongs to.
    memory_path = project_root / config["paths"]["wsdream_memory_path"]
    controller = AgentController(
        backend=backend,
        attribute_cols=WSDREAM_ATTRIBUTE_COLUMNS,
        memory_path=memory_path,
        tool_menu=tuple(config["agent"]["tool_menu"]),
        k_memory=config["agent"]["k_memory"],
    )

    results_dir = project_root / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    output_name = (
        "wsdream_demo_results.csv"
        if args.demo_backend
        else "wsdream_stable_results.csv"
    )
    output_csv = results_dir / output_name

    print(f"\nRunning (resumable, writing incrementally to {output_csv}) ...")
    storage = CsvStorage(
        csv_path=output_csv,
        key_columns=["task_key", "pool_seed", "condition"],
        fieldnames=STABLE_RESULT_FIELDS
    )
    results = run_stable_protocol(
        norm_pool,
        WSDREAM_ATTRIBUTE_COLUMNS,
        storage,
        controller,
        n_pools=n_pools,
        pool_size=pool_size,
        base_seed=wcfg["base_seed"],
        is_synthetic_data=False,
        task_profiles=WSDREAM_TASK_PROFILES,
        held_out_tasks=[],
        global_fixed_weights=WSDREAM_GLOBAL_FIXED_WEIGHTS,
        lookup_table=WSDREAM_TASK_LOOKUP_TABLE,
        lookup_weights_fn=get_wsdream_lookup_weights,
    )
    print(f"\nDone. {len(results)} total trial rows in {output_csv}")
    if args.demo_backend:
        print(
            "Demo output is intentionally excluded from Table 5.4. "
            "Run again with a configured live LLM backend for reportable results."
        )
    else:
        print(
            "Next: python scripts/06_generate_report.py "
            "(picks this file up automatically for Table 5.4)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
