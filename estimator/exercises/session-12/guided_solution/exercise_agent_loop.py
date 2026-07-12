"""The hand-written agent loop — EXERCISE (build it milestone by milestone).

This is the part you rebuild yourself. The reference solution is
``app/generation/agentic/agent_loop.py`` — do NOT import it; compare only after.

WHY BY HAND (do not "fix" this to use LLMWrapper): every other LLM call in the repo
goes through LLMWrapper (LiteLLM + Instructor). Here we drive the raw OpenAI
**Responses API** (``client.responses.create`` / ``.parse``) on purpose, because the
whole point is to SEE the reason→act→observe loop and capture every step in a trace.

────────────────────────────────────────────────────────────────────────────────
HOW TO USE THIS FILE — incremental milestones you can RUN after each one
────────────────────────────────────────────────────────────────────────────────
The loop is gated by four ``MILESTONE_*`` flags below (all start False). With every
flag False the file already RUNS: it does one API round-trip and PRINTS the tool
calls the model wants — so you immediately see the agent *deciding*. Implement one
TODO, flip its flag to True, re-run, and watch the interaction grow:

  Milestone 1 (nothing to do — given): run it. The model reads the transcript and
              asks to call `search_budgets` N times. You see the calls + args. STOP.
  Milestone 2 (TODO 2 + flip MILESTONE_2_EXECUTE_TOOLS): actually run the tools for
              that first turn, record the observations, then STOP. You see one full
              reason→act→observe turn.
  Milestone 3 (TODO 3 + flip MILESTONE_3_CHAIN): feed the outputs back with
              previous_response_id and loop. Now the agent runs many turns until it
              stops asking for tools. You see the whole conversation.
  Milestone 4 (TODO 4 + flip MILESTONE_4_FINAL_PARSE): one final responses.parse to
              turn the accumulated context into a validated AgentEstimate.

Run each milestone (from ``estimator/``) with the offline stub — no DB needed:

    DATABASE_URL='postgresql+psycopg://estimator:estimator@localhost:5433/estimator' \\
    REDIS_URL='redis://localhost:6379' \\
    uv run python exercises/session-12/exercise_run_agent_s12.py \\
        exercises/session-12/sample_transcript_simple.txt --model gpt-5-mini

Add ``--real`` to use the real S9/S10 retrieve() pipeline instead of the stub.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from exercise_agent_schemas import (
    AgentEstimate,
    AgentRunResult,
    AgentStep,
    AgentTrace,
)
from exercise_agent_tools import (
    TOOL_SCHEMAS,
    RetrievalBackend,
    dispatch_tool,
    stub_retrieval_backend,
)

log = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# MILESTONE FLAGS — flip each to True as you complete the matching TODO, so the
# file stays runnable at every stage and you can watch the behaviour grow.
# ─────────────────────────────────────────────────────────────────────────────
MILESTONE_2_EXECUTE_TOOLS = True  # TODO 2: run the tools for a turn
MILESTONE_3_CHAIN = True  # TODO 3: chain previous_response_id and loop
MILESTONE_4_FINAL_PARSE = True  # TODO 5: final responses.parse -> AgentEstimate


SYSTEM_PROMPT = """\
You are an estimation agent for a software consultancy. You receive the raw \
transcript of a discovery meeting and must produce a grounded effort estimate in \
engineer-hours.

Method — follow it step by step:
1. Read the transcript and DECOMPOSE the project into its distinct components \
(for example: a business backend, an ERP integration, a mobile app, an analytics \
dashboard). Real projects usually have several.
2. For EACH component, call `search_budgets` with a focused, component-specific \
query to retrieve how much analogous work has cost historically, in engineer-hours. \
Do one search per component — do not try to cover the whole project in a single \
query.
3. Once you have reference hours for every component, call `calculate_estimate` \
with all the components and their reference amounts to get a partial-and-total \
breakdown.
4. Call `validate_estimate` as the LAST tool step and fix anything it flags \
(e.g. a component with no historical reference — search again for it).
5. When you are satisfied, stop calling tools. You will then be asked to return the \
final structured estimate.

You have exactly these tools: `search_budgets`, `calculate_estimate`, \
`validate_estimate`. Ground your numbers in what `search_budgets` returns; when you \
must assume something the transcript did not specify, record it as an assumption.\
"""

FINAL_INSTRUCTION = (
    "Return the final structured estimate now, consolidating the components you "
    "costed. Set total_hours to the sum of the components, list the assumptions you "
    "made, and choose a confidence level reflecting how well the historical budgets "
    "matched the requested work."
)


# ─────────────────────────────────────────────────────────────────────────────
# GIVEN helpers — you do not need to change these.
# ─────────────────────────────────────────────────────────────────────────────
def _extract_reasoning_summary(output: list[Any]) -> str | None:
    """Concatenate the reasoning-summary text emitted in one turn, if any.

    The Responses API surfaces a summary only when the call passes
    ``reasoning={"summary": "auto"}``; even then it may be empty for cheap efforts.
    """
    parts: list[str] = []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = getattr(summary, "text", None)
            if text:
                parts.append(text)
    return " ".join(parts) if parts else None


def _function_calls(output: list[Any]) -> list[Any]:
    """Return the ``function_call`` items in a response's ``output`` list.

    The Responses API interleaves ``reasoning`` items and ``function_call`` items in
    ``response.output`` and then STOPS, waiting for us to answer each call.
    """
    return [item for item in output if getattr(item, "type", None) == "function_call"]


def _print_turn_header(iteration: int, output: list[Any], calls: list[Any]) -> None:
    """Verbose console view of ONE turn — this is how you SEE the agent think/act."""
    print(f"\n{'─' * 70}")
    print(f"ROUND {iteration}  —  model returned {len(calls)} tool call(s)")
    summary = _extract_reasoning_summary(output)
    if summary:
        print(f"  reasoning: {summary[:300]}")
    for call in calls:
        name = getattr(call, "name", "unknown")
        print(f"  ↳ wants:   {name}({call.arguments})")
    if not calls:
        print("  (no tool calls — the agent is done and ready to give its answer)")
    print("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# THE LOOP — this is what you build.
# ─────────────────────────────────────────────────────────────────────────────
async def run_exercise_estimation_agent(
    transcript: str,
    *,
    client: Any,
    model: str,
    reasoning_effort: str = "medium",
    max_iterations: int = 10,
    retrieval_backend: RetrievalBackend | None = None,
) -> AgentRunResult:
    """Run the manual agent loop over a transcript and return estimate + trace.

    ``client`` is an ``AsyncOpenAI`` instance. ``retrieval_backend`` overrides how
    ``search_budgets`` finds budgets (defaults to the offline stub for this exercise).
    """
    backend = retrieval_backend or stub_retrieval_backend
    trace = AgentTrace()
    step_no = 0
    stopped_reason: str = "completed"

    # ── STEP 3 (given): kick off the loop. We pass the transcript as the user input
    # and the tool schemas. gpt-5 will emit reasoning + function_call items and STOP.
    log.info("agent_run_start", model=model, effort=reasoning_effort)
    response = await client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": transcript}],
        tools=TOOL_SCHEMAS,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        store=True,
    )
    iterations = 1

    while True:
        # ── TODO 1 (given here so Milestone 1 runs): find the function_call items
        # the model wants us to execute. HINT: use the _function_calls() helper.
        calls = _function_calls(response.output)

        # Verbose view so you can watch the interaction (given).
        _print_turn_header(iterations, response.output, calls)

        # Natural stop: a turn with no tool calls means the agent is done.
        if not calls:
            break
        if iterations >= max_iterations:
            stopped_reason = "max_iterations"
            log.warning("agent_max_iterations_reached", iterations=iterations)
            break

        # ─────────────────────────────────────────────────────────────────────
        # MILESTONE 1 GATE: until you implement TODO 2, stop here after showing
        # what the model wanted. Flip MILESTONE_2_EXECUTE_TOOLS to True to go on.
        # ─────────────────────────────────────────────────────────────────────
        if not MILESTONE_2_EXECUTE_TOOLS:
            print(
                "\n[Milestone 1 reached] The agent decided which tools to call.\n"
                "Implement TODO 2 and set MILESTONE_2_EXECUTE_TOOLS = True to run them."
            )
            stopped_reason = "scaffold_incomplete"
            break

        # gpt-5 reasons ONCE per turn even when it emits several parallel tool calls,
        # so the reasoning summary belongs to the turn. Attach it to the first step.
        reasoning_summary = _extract_reasoning_summary(response.output)
        first_step_in_turn = step_no + 1
        tool_outputs: list[dict[str, Any]] = []

        for call in calls:
            step_no += 1
            step_reasoning = (
                reasoning_summary
                if step_no == first_step_in_turn
                else f"(parallel tool call in the same turn as STEP {first_step_in_turn})"
            )
            name = getattr(call, "name", "unknown")

            # ── TODO 2 (DONE): EXECUTE this tool call.
            #   a) call.arguments is a JSON *string* — parse it, guarding against a
            #      malformed payload so a bad call becomes an error, never a crash.
            #   b) On success, dispatch to the tool; guard that too and return the
            #      error as the result so the model can self-correct next turn.
            try:
                raw_args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raw_args = {}
                result = {"error": f"arguments were not valid JSON: {exc}"}
            else:
                try:
                    result = await dispatch_tool(name, raw_args, backend=backend)
                except Exception as exc:  # noqa: BLE001 — return the error so the model self-corrects.
                    log.warning("agent_tool_error", tool=name, error=str(exc)[:200])
                    result = {"error": f"{type(exc).__name__}: {exc}"}

            # observation = a short human-readable line for the trace (given).
            observation = result.get("summary") or result.get("error") or json.dumps(result)[:200]
            print(f"  → {name} observation: {observation}")

            # ── TODO 3a (DONE): record this step in the trace (the deliverable).
            trace.steps.append(
                AgentStep(
                    step=step_no,
                    reasoning_summary=step_reasoning,
                    tool=name,
                    tool_args=raw_args,
                    observation=observation,
                )
            )

            # ── TODO 3b (DONE): build the function_call_output the API expects back.
            # It MUST echo the same call_id so the server can match it to the call.
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        # ─────────────────────────────────────────────────────────────────────
        # MILESTONE 2 GATE: until you implement TODO 4 (chaining), stop after this
        # first turn. Flip MILESTONE_3_CHAIN to True to loop over many turns.
        # ─────────────────────────────────────────────────────────────────────
        if not MILESTONE_3_CHAIN:
            print(
                "\n[Milestone 2 reached] Ran the tools for ONE turn.\n"
                "Implement TODO 4 (chaining) and set MILESTONE_3_CHAIN = True to loop."
            )
            stopped_reason = "scaffold_incomplete"
            break

        # ── TODO 4 (DONE): CONTINUE the conversation. Re-call the Responses API,
        # chaining off the previous response so the server keeps the prior reasoning /
        # function-call ordering. Pass ONLY the new tool_outputs as input.
        response = await client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOL_SCHEMAS,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            store=True,
        )
        iterations += 1

    # ── TODO 5: produce the FINAL structured estimate.
    # After the loop, one responses.parse turns the accumulated context into a
    # validated AgentEstimate. Only do it when we stopped naturally.
    estimate: AgentEstimate | None = None
    if MILESTONE_4_FINAL_PARSE and stopped_reason not in ("max_iterations", "scaffold_incomplete"):
        try:
            # ── TODO 5 (DONE): one final parse. Chain off the last response and ask
            # for the structured AgentEstimate; text_format makes the API validate it.
            parsed = await client.responses.parse(
                model=model,
                previous_response_id=response.id,
                input=[{"role": "user", "content": FINAL_INSTRUCTION}],
                text_format=AgentEstimate,
                store=True,
            )
            estimate = parsed.output_parsed
            iterations += 1
        except Exception as exc:  # noqa: BLE001 — a failed final parse is a stop reason, not a crash.
            log.error("agent_final_parse_failed", error=str(exc)[:300])
            stopped_reason = "no_final_estimate"

    if estimate is None and stopped_reason == "completed":
        stopped_reason = "no_final_estimate"

    log.info(
        "agent_run_done",
        iterations=iterations,
        steps=len(trace.steps),
        stopped_reason=stopped_reason,
        total_hours=(estimate.total_hours if estimate else None),
    )
    return AgentRunResult(
        estimate=estimate,
        trace=trace,
        iterations=iterations,
        stopped_reason=stopped_reason,  # type: ignore[arg-type]
    )
