# estimator-cag

**Author:** Ferran (F.F.V)

Estimator CAG is a FastAPI service for generating software project estimates from meeting transcripts.


## Launch the project

### Local execution (recommended)

**Step 1** — Start Redis (required for caching):

```bash
docker-compose up redis
```

**Step 2** — Start the FastAPI backend:

```bash
make server
```

**Step 3** — Start the UI:

```bash
make ui
```

The API is available at `http://localhost:8000`. The UI is at `http://localhost:8501`.

### With Docker

Starts the API and Redis together (no separate Redis step needed):

Note: This may not be working fully and need some fix to make it work with the UI on docker

```bash
make start-docker
```

To stop:

```bash
make stop-docker
```

## Streamlit UI

```bash
make ui
```

Starts the conversational UI at `http://localhost:8501` against [`streamlit_app.py`](streamlit_app.py). It drives the modern sessions endpoints (`POST /sessions`, `POST /sessions/{id}/estimate`, `GET /sessions/{id}`), supports PDF/DOCX/TXT/MD attachments, shows the live `project_metadata` (memoria) and the windowed `history` (historial) side by side in the sidebar, and exposes a "Nueva conversación" button to reset the state.

Override the backend with the `API_URL` env var (default `http://localhost:8000/api/v1`):

```bash
API_URL=http://custom-api:8000/api/v1 streamlit run streamlit_app.py
```

> The previous stateless form UI (`streamlit_app_form.py`, `make ui-form`) is **deprecated and will be removed**. Use `make ui` exclusively from now on.

## LLM Provider Selection

Every estimation goes through [LiteLLM](https://docs.litellm.ai/) wrapped by `LiteLLMMWrapperService`. The wrapper handles load-balancing, automatic fallback from `PRIMARY_MODEL` to `FALLBACK_MODEL`, Redis-backed exact-match caching, and cost/usage accounting in a single place.

```env
LLM_PROVIDER=lite_llm
PRIMARY_MODEL=openai/gpt-4o-mini
FALLBACK_MODEL=anthropic/claude-haiku-4-5
```

Mixing providers across `PRIMARY_MODEL` / `FALLBACK_MODEL` is supported (OpenAI ↔ Anthropic) — LiteLLM normalises responses and the wrapper turns the result into an Instructor-validated Pydantic model.

> **Legacy paths** — the repository still contains `app/services/llm_service.py`, `app/services/open_ai_service.py`, and `app/services/anthropic_service.py`, plus the SSE endpoint `POST /api/v1/estimate/stream` that calls into them. These are kept for backwards compatibility with the Step-3 streaming UI and are **not** part of the supported flow documented below. New work should target `EstimationService` + `LiteLLMMWrapperService` only.

## Conversational sessions and attachments

The service exposes a session-scoped, multipart endpoint for multi-turn use:

```
POST /api/v1/sessions                          → { "session_id": "<uuid4>" }
POST /api/v1/sessions/{session_id}/estimate    (multipart/form-data)
    transcript:      str          (required, 20–80 000 chars)
    project_type:    str          (web_saas | mobile_app | internal_tool | data_pipeline)
    detail_level:    str          (summary | medium | detailed)
    output_format:   str          (phases_table | line_items | narrative)
    attachments:     file[]       (optional; .pdf, .docx, .txt, .md)
```

### Attachments — Path B: local extraction (chosen)

PDFs are parsed with [`pypdf`](https://pypdf.readthedocs.io/) and Word documents with [`python-docx`](https://python-docx.readthedocs.io/). Extracted text is concatenated to the transcript with a clear `--- attachment: <filename> ---` separator before the prompt is rendered. The rest of the pipeline (input guardrails → exact cache → semantic cache → Instructor structured output → output guardrail → cache writes) runs unchanged.

**Why Path B over Path A (provider Files API):**

- **Provider-independent.** The extracted text flows through the same `EstimationService` + `LiteLLMMWrapperService` path used for plain transcripts, so the OpenAI ↔ Anthropic fallback configured in LiteLLM keeps working. Path A's OpenAI `file_id` references would not survive a fallback to Anthropic.
- **RAG-ready.** Once the text is in our hands we can chunk, embed, and store it — which is the foundation for Module 3 (retrieval-augmented context). Path A would have kept the document opaque inside the provider.
- **Deterministic.** The same file always yields the same text, so the exact-match cache hit rate stays high. Path A relies on the provider's parser, which can change silently.

Trade-offs we accept:

- Extraction is best-effort: scanned PDFs without an OCR layer come back as empty strings. We log a warning; the LLM call still runs but obviously can't reason about the missing content.
- Large attachments can push the augmented description past the 80 000-character `description` limit. Today this fails the Pydantic validator on the augmented request; a future iteration should truncate or summarize before submission.
- Word docs with embedded images and tables are extracted as flat paragraph text — formatting is lost. Acceptable for estimation transcripts.

Implementation: [`app/services/attachments_service.py`](app/services/attachments_service.py) (extractors) and [`app/services/estimation_service.py`](app/services/estimation_service.py) (`estimate_with_attachments`).

### Project metadata (Step 4 — LLM extractor)

After every turn of `POST /sessions/{session_id}/estimate`, the server runs a small extractor call against the LLM to refresh the session's `ProjectMetadata` (project name, assumed team size, mentioned technologies, agreed scope). The refreshed metadata is injected into the **next** turn's system prompt under a `<project_metadata>` block so the model can build on what previous turns established.

**Why LLM extractor instead of regex heuristics:**

- **Language and style coverage.** Transcripts arrive in Spanish and English and mix capitalisation, abbreviations, and informal phrasing. A heuristic that catches one style misses the others; the model normalises across both for free.
- **Reuse, not new infrastructure.** The codebase already runs every estimation through [Instructor](https://python.useinstructor.com/) with `response_model=` for structured output. Pointing the same wrapper at `response_model=ProjectMetadata` keeps the surface area small and the failure modes consistent (Pydantic validators, automatic re-prompts).
- **Marginal cost.** The extractor prompt is small (current metadata JSON + last user transcript + last assistant summary) and runs against the cheap primary model (`gpt-4o-mini` by default). On a typical turn it adds a few hundred prompt tokens — negligible next to the estimation call itself.

What we lose in return: one extra LLM call per turn (~200–500 ms latency, fractions of a cent) and a non-zero rate of "no useful update" turns. We accept this because extractor failures are isolated — [`extract_project_metadata`](app/services/metadata_extractor_service.py) returns the previous metadata unchanged on any error, so the user-facing estimation never breaks because metadata extraction did.

A defensive merge in code (union of `mentioned_technologies`, lowercase canonical form) guarantees the cumulative list cannot shrink between turns even if the LLM omits earlier entries.

Implementation: [`app/services/metadata_extractor_service.py`](app/services/metadata_extractor_service.py), [`app/prompts/estimation/v1/system.j2`](app/prompts/estimation/v1/system.j2) (`<project_metadata>` block), and [`app/prompts/loader.py`](app/prompts/loader.py) (template wiring).

## Health check

```bash
make health
```

## Run a sample estimation request

```bash
make estimate
```

Sends a `POST` to `http://localhost:8000/api/v1/estimate` using the description from:

- `app/data/samples/sample_request_transcription.md`

Override the file:

```bash
make estimate TRANSCRIPTION_FILE=app/data/another_transcription.md
```

## Available Make targets

| Target | Description |
|---|---|
| `make server` | Start the FastAPI backend with auto-reload |
| `make ui` | Start the conversational Streamlit UI |
| `make test` | Run the test suite |
| `make health` | Check `/health` endpoint |
| `make estimate` | Send a sample request |
| `make start-docker` | Build and start API + Redis with Docker Compose |
| `make stop-docker` | Stop Docker Compose services |
| `make stop` | Kill local uvicorn process |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/api/v1/estimate` | Single-turn estimation (stateless, JSON body) |
| `POST` | `/api/v1/sessions` | Create a session — returns `{ "session_id": "<uuid4>" }` |
| `POST` | `/api/v1/sessions/{session_id}/estimate` | Multi-turn estimation with optional attachments (multipart) |
| `GET` | `/api/v1/sessions/{session_id}` | Inspect a session's history + project_metadata (audit) |

### Stateless request body (`POST /api/v1/estimate`)

```json
{
  "description": "Meeting transcript or project requirements text (min 20 chars)",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table"
}
```

### Multi-turn request body (`POST /api/v1/sessions/{session_id}/estimate`, multipart/form-data)

| Field | Type | Notes |
|---|---|---|
| `transcript` | string | Required, 20–80 000 chars |
| `project_type` | string | `web_saas` · `mobile_app` · `internal_tool` · `data_pipeline` |
| `detail_level` | string | `summary` · `medium` · `detailed` |
| `output_format` | string | `phases_table` · `line_items` · `narrative` |
| `attachments` | file[] | Optional; `.pdf`, `.docx`, `.txt`, `.md` |

Response includes the standard `EstimationResponse` alongside the freshly-refreshed `project_metadata` and the current `completed_turns` count, so the client can render the new state without a second round-trip.

## Project structure

```
streamlit_app.py                       # Conversational UI — supported
streamlit_app_form.py                  # Stateless form UI — deprecated, to be removed
app/
  main.py                              # FastAPI entrypoint
  config.py                            # Settings (pydantic-settings, .env)
  routers/
    estimations_route.py               # POST /api/v1/estimate (stateless)
    sessions_route.py                  # POST /sessions, POST /sessions/{id}/estimate, GET /sessions/{id}
  services/
    estimation_service.py              # Pipeline orchestrator (the single entry point)
    litellm_wrapper_service.py         # LiteLLM + Instructor + cost/cache/fallback
    sessions.py                        # ConversationHistory, ProjectMetadata, Session + registry
    attachments_service.py             # Path B: pypdf / python-docx text extraction
    metadata_extractor_service.py      # Post-turn ProjectMetadata extractor (Step 4)
    cache_service.py                   # Redis exact-match cache (SHA-256 keyed)
  cache/
    semantic.py                        # Redis Stack / RediSearch semantic cache
  guardrails/
    input.py                           # Moderation + injection + PII heuristics
    output.py                          # enforce_scope_response (low-confidence filter)
  prompts/
    loader.py                          # render_estimation_prompt(..., project_metadata=...)
    estimation/v1/                     # system.j2 (+ <project_metadata> block), user.j2, examples.j2
  schemas/
    estimation.py                      # EstimationRequest/Response, EstimationResult, enums
    llm_io.py                          # TokenUsage, provider info, …
tests/                                 # pytest suite (incl. tests/test_sessions_endpoint.py)
```

## Request flow — stateless single-turn

`POST /api/v1/estimate` runs the full pipeline atomically. Both caches participate; on a hit, no LLM call is made.

```mermaid
flowchart LR
    Client(["Client"])
    Router["estimations_route.py"]
    Service["EstimationService.estimate()"]
    InputGR["Input guardrails"]
    Exact[("Redis exact cache")]
    Semantic[("Redis Stack semantic cache")]
    Render["render_estimation_prompt\n(Jinja2: system.j2 + user.j2)"]
    Wrapper["LiteLLMMWrapperService\ncomplete_structured\n(Instructor + Pydantic)"]
    LLM[("OpenAI / Anthropic\nvia LiteLLM Router")]
    OutGR["Output guardrail"]

    Client -->|EstimationRequest| Router
    Router --> Service
    Service --> InputGR
    InputGR --> Exact
    Exact -- miss --> Semantic
    Exact -- hit --> Client
    Semantic -- miss --> Render
    Semantic -- hit --> Client
    Render --> Wrapper
    Wrapper --> LLM
    LLM --> Wrapper
    Wrapper --> OutGR
    OutGR -->|"write back\n(exact + semantic)"| Exact
    OutGR -->|EstimationResponse| Client
```

## Request flow — multi-turn session

`POST /api/v1/sessions/{session_id}/estimate` adds three things on top of the stateless flow: attachment extraction, history-aware prompting, and post-turn metadata extraction. Caches are skipped on multi-turn calls because every turn carries a unique prior context.

```mermaid
flowchart TB
    Client(["Client"])
    SessRouter["sessions_route.py"]
    Registry[("Session registry\nprocess-local dict\nConversationHistory + ProjectMetadata")]
    Attach["attachments_service\nextract_text\npypdf / python-docx"]
    Service["EstimationService.estimate_with_attachments\n(augments description, builds messages\nfrom history.to_messages_list)"]
    Wrapper["LiteLLMMWrapperService\ncomplete_structured_messages"]
    LLM[("OpenAI / Anthropic\nvia LiteLLM Router")]
    Extractor["metadata_extractor_service\nextract_project_metadata\n(Instructor: response_model=ProjectMetadata)"]

    Client -->|"multipart\ntranscript + attachments"| SessRouter
    SessRouter -->|"get_session(id)"| Registry
    Registry -->|"history + project_metadata"| SessRouter
    SessRouter -->|"raw bytes"| Attach
    Attach -->|"--- attachment: <name> ---\n<extracted text>"| SessRouter
    SessRouter -->|"request + attachments\n+ history + project_metadata"| Service
    Service -->|"messages = [system(project_metadata),\n  history pairs,\n  current user]"| Wrapper
    Wrapper --> LLM
    LLM --> Wrapper
    Wrapper -->|"EstimationResult"| Service
    Service -->|"EstimationResponse"| SessRouter
    SessRouter -->|"history.add(user); history.add(assistant)"| Registry
    SessRouter -->|"current metadata + transcript + summary"| Extractor
    Extractor -->|"complete_structured(ProjectMetadata)"| Wrapper
    Extractor -->|"merged ProjectMetadata"| Registry
    SessRouter -->|"SessionEstimateResponse\n(result + project_metadata + completed_turns)"| Client
```

Two invariants worth holding in mind while reading the flow:

- The system prompt is **regenerated from `ProjectMetadata` on every turn** via `render_estimation_prompt(..., project_metadata=...)`. It is never persisted in `ConversationHistory`.
- `ProjectMetadata` survives the sliding-window truncation that `ConversationHistory` performs after each turn. The metadata block in `system.j2` ends with a "treat as established facts" instruction so the LLM keeps respecting facts whose originating turns have rolled out of the window.
