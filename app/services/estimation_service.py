"""Pipeline orchestrator. Glue between guardrails, caches, prompt rendering and
the LLM wrapper. The router holds none of this logic — its only job is to
translate HTTP errors.

Pipeline (Session 4, final):

    1. Input guardrails (moderation + prompt injection + PII heuristics)
    2. Exact-match cache lookup  → return cached=True on hit
    3. Semantic cache lookup     → return cached=True on hit
    4. Render the versioned prompt
    5. LLM call via Instructor with response_model=EstimationResult
    6. Output guardrail (enforce_scope_response, filter policy)
    7. Write to BOTH caches (exact + semantic)
    8. Return EstimationResponse with cached=False

Order rationale: guardrails go before any cache because a malicious or PII
description should never be served from cache. The exact-match cache goes
before the semantic cache because it's the cheapest (no embedding call). The
semantic cache write happens AFTER output validation so we never cache failed
estimations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

from app.cache.semantic import EstimationSemanticCache
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import (
    AttachmentExtraction,
    EstimationResult,
    EstimationRequest,
    EstimationResponse,
)
from app.services.attachments_service import extract_text
from app.services.cache_service import EstimationCache
from app.services.litellm_wrapper_service import LiteLLMMWrapperService
from app.services.sessions import ConversationHistory, ProjectMetadata

ATTACHMENT_SEPARATOR_TEMPLATE = "\n\n--- attachment: {filename} ---\n{text}"

log = structlog.get_logger()


def _metadata_fingerprint(metadata: ProjectMetadata | None) -> dict[str, Any]:
    """Stable, JSON-serialisable view of metadata for cache keying."""
    if metadata is None:
        return {}
    techs = sorted({t.strip().lower() for t in metadata.mentioned_technologies if t})
    return {
        "project_name": metadata.project_name or None,
        "assumed_team_size": metadata.assumed_team_size or None,
        "mentioned_technologies": techs,
        "agreed_scope": metadata.agreed_scope or None,
    }


def _has_metadata(metadata: ProjectMetadata | None) -> bool:
    if metadata is None:
        return False
    return bool(
        metadata.project_name
        or metadata.assumed_team_size
        or metadata.mentioned_technologies
        or metadata.agreed_scope
    )


def _exact_cache_key(
    request: EstimationRequest,
    prompt_version: str,
    model: str,
    project_metadata: ProjectMetadata | None = None,
) -> str:
    """Deterministic SHA-256 key over the typed request + prompt_version + model.

    ``project_metadata`` is part of the key when present: two sessions with
    identical descriptions but different known facts produce different
    rendered prompts, so they must not share a cache entry.
    """
    payload = json.dumps(
        {
            "description": request.description,
            "project_type": request.project_type.value,
            "detail_level": request.detail_level.value,
            "output_format": request.output_format.value,
            "prompt_version": prompt_version,
            "model": model,
            "project_metadata": _metadata_fingerprint(project_metadata),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"estimation:v2:{digest}"


class EstimationService:
    """Single entry point for the structured estimation pipeline."""

    def __init__(
        self,
        *,
        llm_wrapper: LiteLLMMWrapperService,
        exact_cache: EstimationCache,
        semantic_cache: EstimationSemanticCache | None = None,
        openai_client: Any | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.openai_client = openai_client
        self.prompt_version = prompt_version

    # Extracts text from attachments and appends it to the description with separators, then calls the normal estimation pipeline with the augmented description.
    def estimate_with_attachments(
        self,
        request: EstimationRequest,
        attachments: list[tuple[str, bytes]],
        project_metadata: ProjectMetadata | None = None,
        history: ConversationHistory | None = None,
    ) -> EstimationResponse:
        """Variant of ``estimate`` that augments the description with text
        extracted from attached documents (Path B — local extraction).

        Each ``(filename, content)`` tuple is parsed by ``attachments_service``
        (PDF/DOCX/TXT/MD) and concatenated to the transcript with a clear
        ``--- attachment: <name> ---`` separator. The rest of the pipeline
        (guardrails, caches, structured-output call) runs unchanged: from its
        point of view it just receives a longer ``description``.

        Returns the same ``EstimationResponse`` as ``estimate``.
        """
        if not attachments:
            return self.estimate(
                request,
                project_metadata=project_metadata,
                history=history,
            )

        parts: list[str] = [request.description]
        extractions: list[AttachmentExtraction] = []
        for filename, content in attachments:
            text = extract_text(filename, content)
            parts.append(
                ATTACHMENT_SEPARATOR_TEMPLATE.format(filename=filename, text=text)
            )
            extractions.append(
                AttachmentExtraction(
                    filename=filename,
                    bytes=len(content),
                    chars=len(text),
                    text=text,
                )
            )

        log.info(
            "estimation_attachments_processed",
            count=len(attachments),
            attachments=[
                {
                    "filename": e.filename,
                    "bytes": e.bytes,
                    "chars": e.chars,
                    "preview": (e.text[:160] + "…") if len(e.text) > 160 else e.text,
                }
                for e in extractions
            ],
        )

        augmented_request = request.model_copy(
            update={"description": "\n".join(parts)}
        )

        log.info(
            "estimation_with_attachments",
            description_cut=augmented_request.description[:200],
            project_type=augmented_request.project_type.value,
            detail_level=augmented_request.detail_level.value,
            output_format=augmented_request.output_format.value
        )   

        # Call the normal estimation pipeline with the augmented description including the attachment content.
        # The caches are keyed off the raw description + metadata, so they do not see the attachment content —
        # this means we get no hits on multi-turn calls (since the prior conversation would not include the attachment content either)
        # and only hits on single-turn calls where the exact same description + metadata was seen before.
        #  This is a tradeoff: including the attachment content in the cache key would allow hits on single-turn calls but poison future
        #  lookups with entries that cannot be hit without attachments; excluding it means we never get a hit on calls with attachments,
        #  but also never risk an incorrect hit.
        response = self.estimate(
            augmented_request,
            project_metadata=project_metadata,
            history=history,
        )
        # Surface the per-file extraction trace to the caller so the UI can
        # show "what the LLM actually saw" for each uploaded document.
        return response.model_copy(update={"attachments": extractions})

    # The main entry point for the estimation pipeline. See ``estimate_with_attachments`` for the variant that supports attached documents.
    def estimate(
        self,
        request: EstimationRequest,
        project_metadata: ProjectMetadata | None = None,
        history: ConversationHistory | None = None,
    ) -> EstimationResponse:
        # 1. Input guardrails — raises InputGuardrailViolation on rejection.
        check_input(request.description, openai_client=self.openai_client)

        # A non-empty conversation history makes every turn unique, so the
        # exact and semantic caches both miss almost always — and a stray hit
        # against an entry recorded under a *different* prior conversation
        # would be wrong. We skip both caches entirely on multi-turn calls
        # and only persist results from stateless ones.
        is_multi_turn = history is not None and len(history) > 0

        # 2. Exact-match cache lookup. ``project_metadata`` is part of the key
        #    when present: a session whose known facts differ from a previous
        #    one must not share a cache entry, since the rendered prompt — and
        #    therefore the answer — would differ.
        cache_key = _exact_cache_key(
            request,
            self.prompt_version,
            self.llm_wrapper.primary_model,
            project_metadata=project_metadata,
        )
        if not is_multi_turn:
            cached = self.exact_cache.get(cache_key)
            if cached:
                log.info(
                    "estimation_cache_hit",
                    kind="exact",
                    key_prefix=cache_key[:24],
                )
                result = EstimationResult.model_validate(cached["result"])
                return EstimationResponse(
                    result=result, prompt_version=self.prompt_version, cached=True
                )

        # 3. Semantic cache lookup. Skipped when project_metadata carries any
        #    fact — the bucket key does not include metadata, so a semantic hit
        #    could return an answer rendered against a different prior context.
        if (
            not is_multi_turn
            and self.semantic_cache is not None
            and not _has_metadata(project_metadata)
        ):
            semantic_hit = self.semantic_cache.lookup(request, self.prompt_version)
            if semantic_hit is not None:
                log.info("estimation_cache_hit", kind="semantic")
                return EstimationResponse(
                    result=semantic_hit,
                    prompt_version=self.prompt_version,
                    cached=True,
                )

        # 4. Render the versioned prompt.
        system_prompt, user_message = render_estimation_prompt(
            request,
            version=self.prompt_version,
            project_metadata=project_metadata,
        )

        # 5. LLM call. Single-turn requests go through the convenience helper;
        #    multi-turn requests build the message array from the session's
        #    sliding-window history (system regenerated, prior pairs, new turn).
        if is_multi_turn:
            messages = history.to_messages_list(system_prompt)
            messages.append({"role": "user", "content": user_message})
            result, meta = self.llm_wrapper.complete_structured_messages(
                messages=messages,
                response_model=EstimationResult,
            )
        else:
            result, meta = self.llm_wrapper.complete_structured(
                system_prompt=system_prompt,
                user_message=user_message,
                response_model=EstimationResult,
            )
        log.info(
            "estimation_generated",
            prompt_version=self.prompt_version,
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
            phases=len(result.phases),
            multi_turn=is_multi_turn,
            history_messages=len(history) if history else 0,
            **meta,
        )

        # 6. Output guardrail (filter): normalises low-confidence answers.
        result = enforce_scope_response(result)

        # 7. Cache the validated payload only on stateless calls — multi-turn
        #    results are tightly coupled to the prior conversation and would
        #    poison future lookups if cached.
        if not is_multi_turn:
            self.exact_cache.set(
                cache_key,
                {
                    "result": result.model_dump(mode="json"),
                    "prompt_version": self.prompt_version,
                },
            )
            if self.semantic_cache is not None and not _has_metadata(project_metadata):
                self.semantic_cache.store(request, result, self.prompt_version)

        # 8. Return.
        return EstimationResponse(
            result=result, prompt_version=self.prompt_version, cached=False,latency_ms=meta.get("latency_ms"), cost_usd=meta.get("cost_usd"),provider=meta.get("provider"), model=meta.get("model")
        )