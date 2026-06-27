#!/usr/bin/env python3
"""Pre-work scope of Session 10: hybrid search + reranking ONLY.

Measures the four baseline configurations of the exercise statement against the
five budget-only golden queries (Q1-Q5), reporting precision@5 (mean over the
queries) and end-to-end retrieval latency (mean over measured runs):

    Config  Search   Reranking
    A       Vector   No
    B       Hybrid   No
    C       Vector   Yes
    D       Hybrid   Yes

This is the focused counterpart of ``eval_retrieval_s10.py`` (which also runs the
live-session techniques E-H and the cross-collection queries Q6-Q8 that are out
of scope here). It reuses the same pure pipeline functions and grading, so the
numbers are directly comparable.

Usage (host on Linux, or inside the container)::

    uv run python scripts/eval_s10_hybrid_rerank.py
    docker compose run --rm estimator python scripts/eval_s10_hybrid_rerank.py

Prerequisite: budgets ingested via ``scripts/query_examples.py``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.s08_common import Stopwatch, require_embedder  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.dependencies import get_reranker  # noqa: E402
from app.generation.rag.retrieval.collections import Collection  # noqa: E402

# Reuse the exact harness primitives so grading + stage wiring stay identical.
from scripts.eval_retrieval_s10 import (  # noqa: E402
    BUDGET_ONLY,
    MEASURED_RUNS,
    NamedConfig,
    _run_once,
    precision_at_k,
    relevant_ids,
    _stages,
)

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"

# The four exercise baselines, budgets only.
CONFIGS: list[NamedConfig] = [
    NamedConfig("A", "Vector / no rerank", _stages(search_mode="vector"), BUDGET_ONLY),
    NamedConfig("B", "Hybrid / no rerank", _stages(search_mode="hybrid"), BUDGET_ONLY),
    NamedConfig("C", "Vector / rerank", _stages(search_mode="vector", rerank=True), BUDGET_ONLY),
    NamedConfig("D", "Hybrid / rerank", _stages(search_mode="hybrid", rerank=True), BUDGET_ONLY),
]

# Only the budget-only queries are in scope (no routing / multi-index here).
BUDGET_QUERY_IDS = {"Q1", "Q2", "Q3", "Q4", "Q5"}


async def main() -> int:
    get_settings()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = [q for q in golden["queries"] if q["id"] in BUDGET_QUERY_IDS]
    k = int(golden.get("k", 5))
    reference_date = date.today()

    embedder = require_embedder()
    print("Warming up the cross-encoder (first load downloads weights)...")
    get_reranker().load()

    results = {cfg.id: {"precisions": [], "latencies_ms": [], "per_query": {}} for cfg in CONFIGS}

    for cfg in CONFIGS:
        for q in queries:
            relevant = relevant_ids(q)
            await _run_once(cfg, q["query"], embedder, reference_date)  # warm-up, discarded
            samples = []
            last = None
            for _ in range(MEASURED_RUNS):
                with Stopwatch() as sw:
                    last = await _run_once(cfg, q["query"], embedder, reference_date)
                samples.append(sw.elapsed_ms)
            precision = precision_at_k(last, relevant, k)
            results[cfg.id]["precisions"].append(precision)
            results[cfg.id]["latencies_ms"].extend(samples)
            results[cfg.id]["per_query"][q["id"]] = precision

    _print_report(results, queries, k)
    return 0


def _print_report(results: dict, queries: list, k: int) -> None:
    print(f"\n## Hybrid + reranking — precision@{k} and latency (Q1-Q5, budgets)\n")
    print(f"| Config | Search | Reranking | Precision@{k} | Latency (ms) |")
    print("| --- | --- | --- | --- | --- |")
    labels = {
        "A": ("Vector", "No"),
        "B": ("Hybrid", "No"),
        "C": ("Vector", "Yes"),
        "D": ("Hybrid", "Yes"),
    }
    for cfg in CONFIGS:
        bucket = results[cfg.id]
        mean_p = statistics.fmean(bucket["precisions"])
        mean_l = statistics.fmean(bucket["latencies_ms"])
        search, rerank = labels[cfg.id]
        print(f"| {cfg.id} | {search} | {rerank} | {mean_p:.2f} | {mean_l:.1f} |")

    print(f"\n### Per-query precision@{k}\n")
    print("| Query | " + " | ".join(cfg.id for cfg in CONFIGS) + " |")
    print("| --- | " + " | ".join("---" for _ in CONFIGS) + " |")
    for q in queries:
        row = [q["id"]] + [f"{results[cfg.id]['per_query'][q['id']]:.2f}" for cfg in CONFIGS]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
