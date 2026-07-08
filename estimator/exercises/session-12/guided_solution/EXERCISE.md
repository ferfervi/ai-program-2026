# Session 12 — build the agent loop by hand (guided milestones)

A self-paced, runnable-at-every-step version of the Session 12 agent. You rebuild the
manual reason→act→observe loop yourself, watching the agent–tool interaction grow one
milestone at a time. Nothing here imports the reference solution
(`app/generation/agentic/`) — diff against it only *after* you finish.

## Files in this folder

| File | Role |
|---|---|
| `exercise_agent_schemas.py` | **Given** — Pydantic models (tool args, trace, result). |
| `exercise_agent_tools.py` | **Given** — flat `strict:true` Responses tool schemas + impls + both retrieval backends (stub / real). |
| `exercise_agent_loop.py` | **You build** — `run_exercise_estimation_agent`, milestone by milestone. |
| `exercise_run_agent_s12.py` | Runner. Imports the exercise files; offline **stub by default**, `--real` for the S9/S10 `retrieve()` pipeline. |
| `reference_retrieval.py` | The offline stub corpus (canned budgets, no DB). |
| `calculate_estimate_skeleton.py` | Original starter for the cost function (already folded into `exercise_agent_tools.py`). |
| `sample_transcript_simple.txt` / `sample_transcript_complex.txt` | Test inputs (1 component / 4 components). |

## How the milestones work

`exercise_agent_loop.py` has three flags at the top, all starting `False`:

```python
MILESTONE_2_EXECUTE_TOOLS = False   # TODO 2: run the tools for a turn
MILESTONE_3_CHAIN         = False   # TODO 3 + 4: chain previous_response_id and loop
MILESTONE_4_FINAL_PARSE   = False   # TODO 5: final responses.parse -> AgentEstimate
```

The file **runs at every stage**. Implement one TODO, flip its flag to `True`, re-run,
and observe more of the interaction. Each gate prints a message telling you what to do
next.

| Milestone | Implement | Flip | What you see when you run |
|---|---|---|---|
| **1** | nothing (given) | — | The agent reads the transcript and **decides** which tools to call (name + args + reasoning summary), then stops. |
| **2** | TODO 2 — parse `call.arguments` with `json.loads` (guarded), then `await dispatch_tool(name, raw_args, backend=backend)` (guarded) | `MILESTONE_2_EXECUTE_TOOLS` | One full **reason → act → observe** turn: the tools actually run and return observations. |
| **3** | TODO 3a (append an `AgentStep` to the trace), TODO 3b (build the `function_call_output` dict), TODO 4 (re-call `responses.create` with `previous_response_id` + only `tool_outputs`, then `iterations += 1`) | `MILESTONE_3_CHAIN` | The **whole multi-turn loop** until the agent stops asking for tools; the trace fills up. |
| **4** | TODO 5 — final `client.responses.parse(..., text_format=AgentEstimate)` | `MILESTONE_4_FINAL_PARSE` | The **final structured estimate** (components, total, confidence, assumptions). |

## Running

From the `estimator/` directory. Offline stub (no DB), cheap model — use this while
building the loop:

```bash
uv run python exercises/session-12/exercise_run_agent_s12.py \
    exercises/session-12/sample_transcript_simple.txt --model gpt-5-mini --effort low
```

Real retrieval (needs the stack up, the task corpus ingested, and the localhost DB
overrides). This is also the shape of the deliverable run:

```bash
DATABASE_URL='postgresql+psycopg://estimator:estimator@localhost:5433/estimator' \
REDIS_URL='redis://localhost:6379' \
uv run python exercises/session-12/exercise_run_agent_s12.py \
    exercises/session-12/sample_transcript_complex.txt --model gpt-5 --effort medium \
    --real --out exercises/session-12/my_trace_complex.txt
```

> The real backend filters to `chunk_type='historical_task'`, so it needs the task
> corpus ingested first: `python scripts/build_task_corpus.py --ingest`.

## Key mechanics the TODOs teach (why they matter)

- **Guard every tool call.** Wrap `json.loads(call.arguments)` and `dispatch_tool(...)`
  in `try/except` and return the error as the observation string. A bad/hallucinated
  argument must become something the model can self-correct from — never a crash.
- **Echo the `call_id`.** Each `function_call_output` must carry the same `call_id` as
  the `function_call` it answers, or the API can't match them.
- **Chain, don't resend.** Re-call with `previous_response_id=response.id` and pass
  **only** the new `tool_outputs` as input. The server keeps the prior reasoning /
  function-call ordering — this sidesteps the gpt-5 reasoning-item ordering pitfalls.
- **Two stop conditions.** Natural stop = a turn with no `function_call`. Safeguard =
  `max_iterations`. Only run the final parse when you stopped naturally.
- **Drive the loop by hand.** Do not delegate chaining to the API's built-in agentic
  behaviour — driving it yourself is what lets you capture the per-step trace, which is
  half the exercise.

## Acceptance criteria (with `sample_transcript_complex.txt`)

- Identifies **more than one component** and makes **more than one** `search_budgets`
  call.
- Calls `calculate_estimate` with the components and their references.
- Terminates on its own (no infinite loop, no mid-run cut).
- Produces a coherent structured estimate.
- The trace shows, per step: reasoning + action + observation.

## Deliverable

The execution trace for `sample_transcript_complex.txt` (write it out with `--out`).

## Compare with the reference (only after you finish)

- `app/generation/agentic/agent_schemas.py`
- `app/generation/agentic/agent_tools.py`
- `app/generation/agentic/agent_loop.py`
- `scripts/run_agent_s12.py`
