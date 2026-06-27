# Context client for the Session 9 RAG estimation pipeline. Talks to the
# per-stage wizard endpoints (POST /v1/estimate/stages/*) and the full
# transcript → estimate endpoint, all guarded by the estimate API key
# (X-API-Key header, injected once via BaseClient#default_headers).
#
# Stateless contract: each stage takes the prior stage's artifacts and returns
# the next one, so the caller (the wizard controller) persists everything and
# can re-run a single stage in isolation.
module EstimatorAi
  class RagEstimateClient < BaseClient
    def initialize(timeout: Rails.application.config.estimator_ai.timeout)
      super(
        timeout: timeout,
        default_headers: { "X-API-Key" => Rails.application.config.estimator_ai.estimate_api_key }
      )
    end

    # Stage 1 — transcript → { "query" => {...}, "search_text" => "..." }.
    def reformulate(transcript:)
      handle_response(json_conn.post("/v1/estimate/stages/reformulate", { transcript: transcript }))
    end

    # Structure-only generation (Session 10): a FREE decomposition of the brief —
    # no retrieval, no sources. Returns the same shape as the grounded generate
    # ({ estimate, fabricated_source_ids, coherent }) so the views parse it
    # unchanged; for a structure, citations are always clean.
    def generate_structure(query:)
      handle_response(json_conn.post("/v1/estimate/stages/structure", { query: query }))
    end

    # Grounded single-stage generation (Session 9, kept for the side-by-side
    # comparison demo; the Session 10 wizard uses #generate_structure instead).
    def generate(context_block:, query:, kept_chunks:, include_hours: true)
      handle_response(json_conn.post("/v1/estimate/stages/generate", {
        context_block: context_block, query: query, kept_chunks: kept_chunks,
        include_hours: include_hours
      }))
    end

    # Session 10 — per-task hours by vector search over the historical task corpus.
    # ``modules`` is [{ name:, tasks: [{ name:, description: }] }]; returns
    # { "tasks" => [TaskHoursEstimate, ...] } with reliability + neighbours.
    def estimate_task_hours(modules:)
      handle_response(json_conn.post("/v1/estimate/tasks/hours", { modules: modules }))
    end

    # Full pipeline (single shot, idempotent). Kept for the "compare against the
    # one-shot path" demo; the wizard itself drives the stages above.
    def from_transcript(transcript:, idempotency_key: nil)
      payload = { transcript: transcript, idempotency_key: idempotency_key }.compact
      handle_response(json_conn.post("/v1/estimate/from-transcript", payload))
    end
  end
end
