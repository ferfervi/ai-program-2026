#!/usr/bin/env python3
"""Session 12 — run YOUR hand-built estimation agent over a transcript (EXERCISE).

Mirror of ``scripts/run_agent_s12.py`` but wired to the exercise files in THIS folder
instead of the reference solution:

    exercise_agent_loop.py     ← the loop you build milestone by milestone
    exercise_agent_tools.py    ← tool schemas + impls + both retrieval backends
    exercise_agent_schemas.py  ← the Pydantic models

Retrieval defaults to the OFFLINE STUB (``reference_retrieval.py``) so you can iterate
on the loop with no database. Pass ``--real`` to wrap the real S9/S10 ``retrieve()``
pipeline instead (needs the stack up + the historical-task corpus ingested, and the
localhost DB overrides).

    # Offline stub (default) — build the loop cheaply:
    DATABASE_URL='postgresql+psycopg://estimator:estimator@localhost:5433/estimator' \\
    REDIS_URL='redis://localhost:6379' \\
    uv run python exercises/session-12/exercise_run_agent_s12.py \\
        exercises/session-12/sample_transcript_simple.txt --model gpt-5-mini

    # Real retrieval, deliverable run on the complex transcript:
    uv run python exercises/session-12/exercise_run_agent_s12.py \\
        exercises/session-12/sample_transcript_complex.txt --model gpt-5 --effort medium \\
        --real --out exercises/session-12/my_trace_complex.txt

(Run from the ``estimator/`` directory.)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# This file lives at estimator/exercises/session-12/. Add estimator/ to the path so
# ``app.*`` imports resolve, and this folder so the sibling exercise_* modules do.
EXERCISE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXERCISE_DIR.parent.parent  # .../estimator
for path in (REPO_ROOT, EXERCISE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.config import get_settings  # noqa: E402
from app.dependencies import get_async_openai_client  # noqa: E402
from exercise_agent_loop import run_exercise_estimation_agent  # noqa: E402
from exercise_agent_schemas import AgentRunResult  # noqa: E402
from exercise_agent_tools import real_retrieval_backend, stub_retrieval_backend  # noqa: E402


def _render(result: AgentRunResult) -> str:
    lines = [
        "=" * 78,
        "AGENT TRACE",
        "=" * 78,
        result.trace.render(),
        "",
        "=" * 78,
        f"FINAL ESTIMATE  (iterations={result.iterations}, stopped={result.stopped_reason})",
        "=" * 78,
    ]
    estimate = result.estimate
    if estimate is None:
        lines.append("(no structured estimate yet — expected until you reach Milestone 4)")
        return "\n".join(lines)

    for component in estimate.components:
        cited = ", ".join(str(c) for c in component.cited_chunk_ids) or "none"
        lines.append(f"  - {component.name}: {component.estimated_hours}h  [sources: {cited}]")
        lines.append(f"      {component.rationale}")
    lines.append("")
    lines.append(f"  TOTAL: {estimate.total_hours}h    confidence: {estimate.confidence}")
    if estimate.assumptions:
        lines.append("  assumptions:")
        for assumption in estimate.assumptions:
            lines.append(f"    · {assumption}")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    client = get_async_openai_client()
    if client is None:
        print(
            "ERROR: OPENAI_API_KEY is not set — the agent needs the OpenAI Responses API.",
            file=sys.stderr,
        )
        return 1

    backend = real_retrieval_backend if args.real else stub_retrieval_backend
    transcript = transcript_path.read_text(encoding="utf-8")

    print(f"transcript : {transcript_path}")
    print(
        f"model      : {args.model}   effort: {args.effort}   backend: "
        f"{'real retrieve() pipeline' if args.real else 'offline stub'}"
    )
    print()

    result = await run_exercise_estimation_agent(
        transcript,
        client=client,
        model=args.model,
        reasoning_effort=args.effort,
        max_iterations=args.max_iterations,
        retrieval_backend=backend,
    )

    rendered = _render(result)
    print(rendered)

    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"\n(trace written to {args.out})")
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run YOUR Session 12 estimation agent (exercise).")
    parser.add_argument("transcript", help="Path to a meeting transcript .txt file.")
    parser.add_argument(
        "--model",
        default=settings.AGENT_MODEL,
        help=f"OpenAI model (default {settings.AGENT_MODEL}).",
    )
    parser.add_argument(
        "--effort",
        default=settings.AGENT_REASONING_EFFORT,
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for the Responses API.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.AGENT_MAX_ITERATIONS,
        help="Loop safeguard: max Responses API round-trips.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Wrap the real S9/S10 retrieve() pipeline (needs the DB + task corpus). "
        "Default is the offline stub.",
    )
    parser.add_argument("--out", help="Write the rendered trace + estimate to this file.")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
