"""Session lifecycle endpoints.

Step 2 of the agent loop: a client opens a session, gets a ``session_id``
(UUID v4), and sends that id back on every subsequent request so the server
can resume the same ``Session`` (history + project metadata).

Step 3 adds ``POST /sessions/{session_id}/estimate`` — multipart endpoint
that accepts a transcript plus optional PDF/Word/text attachments. Path B
(local extraction) lives in ``attachments_service``.
"""

from __future__ import annotations

import re
import uuid

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from instructor.core import InstructorRetryException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.dependencies import get_estimation_service, get_llm_wrapper
from app.guardrails.input import InputGuardrailViolation
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)
from app.services.attachments_service import AttachmentExtractionError
from app.services.estimation_service import EstimationService
from app.services.metadata_extractor_service import extract_project_metadata
from app.services.sessions import (
    ConversationHistory,
    ProjectMetadata,
    get_or_create_session,
    get_session,
)

log = structlog.get_logger()

router = APIRouter(tags=["sessions"])


class SessionCreatedResponse(BaseModel):
    session_id: str = Field(
        description="Opaque identifier (UUID v4). Send it back as the "
        "`session_id` field of every subsequent request to keep state."
    )


class _HistoryEntry(BaseModel):
    role: str
    content: str


class SessionStateResponse(BaseModel):
    """Inspectable snapshot of a session — used for audit and tests.

    Exposes both layers of conversational state side by side:
    - ``history``: the sliding window of (user, assistant) messages.
    - ``project_metadata``: the distilled facts that survive truncation.
    """

    session_id: str
    completed_turns: int = Field(
        description="Number of fully closed (user, assistant) pairs in the window."
    )
    max_turns: int = Field(description="Sliding-window capacity in pairs.")
    history: list[_HistoryEntry]
    project_metadata: ProjectMetadata


class SessionEstimateResponse(BaseModel):
    """Response of POST /sessions/{session_id}/estimate.

    Wraps the standard ``EstimationResponse`` and also exposes the
    project_metadata that the extractor produced **after** this turn —
    so the caller sees what the system now believes about the project
    without having to re-fetch the session.
    """

    estimation: EstimationResponse
    project_metadata: ProjectMetadata
    completed_turns: int


def _history_to_entries(history: ConversationHistory) -> list[_HistoryEntry]:
    return [
        _HistoryEntry(role=m.role, content=m.content) for m in history.messages
    ]


_VALIDATION_FAILURE_HINT = (
    "Try providing more concrete details — explicit scope items, team "
    "composition, deadlines, or a budget — so the model can size each "
    "phase before computing the totals."
)


def _summarise_validation_failure(exc: InstructorRetryException) -> str:
    """Pull the last 'Value error, …' line out of an InstructorRetryException.

    The exception's string contains every failed retry attempt with its
    Pydantic ``ValueError`` payload; the most recent one is the most
    useful to surface to the caller.
    """
    text = str(exc)
    matches = re.findall(r"Value error,\s*([^\[]+?)(?:\s*\[type=value_error)", text)
    if matches:
        return matches[-1].strip().rstrip(".;,")
    # Fallback: a non-Value-error validation issue (missing field, wrong type).
    matches = re.findall(r"\d+\s+validation errors?\s+for\s+(\w+)", text)
    if matches:
        return f"the model did not produce a valid {matches[-1]} payload"
    return "the model could not produce a self-consistent estimation"


@router.post("/sessions", response_model=SessionCreatedResponse, status_code=201)
def create_session() -> SessionCreatedResponse:
    session_id = str(uuid.uuid4())
    get_or_create_session(session_id, max_turns=get_settings().SESSION_MAX_TURNS)
    log.info("session_endpoint_created", session_id=session_id)
    return SessionCreatedResponse(session_id=session_id)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionStateResponse,
)
def get_session_state(session_id: str) -> SessionStateResponse:
    """Inspect a session's history + project metadata.

    Auditability hook: lets operators answer "why did the LLM assume X
    here?" by surfacing the distilled facts the system was carrying into
    each turn. Also drives the multi-turn integration tests in Step 7.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found.",
        )
    return SessionStateResponse(
        session_id=session.session_id,
        completed_turns=session.history.completed_turns,
        max_turns=session.history.max_turns,
        history=_history_to_entries(session.history),
        project_metadata=session.metadata,
    )


@router.post(
    "/sessions/{session_id}/estimate",
    response_model=SessionEstimateResponse,
)
async def estimate_in_session(
    session_id: str,
    transcript: str = Form(..., min_length=20, max_length=80000),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    attachments: list[UploadFile] | None = File(default=None),
    service: EstimationService = Depends(get_estimation_service),
) -> SessionEstimateResponse:
    """Multi-turn estimation with optional attachments (Path B).

    Attachments (PDF / DOCX / TXT / MD) are read into memory, their text is
    extracted locally, and the result is concatenated into the transcript
    before going through the normal estimation pipeline. The session's
    conversation history is updated with the turn.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found. Call POST /sessions first.",
        )

    raw_attachments: list[tuple[str, bytes]] = []
    if attachments:
        for upload in attachments:
            content = await upload.read()
            raw_attachments.append((upload.filename or "attachment", content))

    request = EstimationRequest(
        description=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        session_id=session_id,
    )

    log.info(
        "session_estimate_request_received",
        session_id=session_id,
        attachments=len(raw_attachments),
        transcript_chars=len(transcript),
        known_facts=bool(
            session.metadata.project_name
            or session.metadata.assumed_team_size
            or session.metadata.mentioned_technologies
            or session.metadata.agreed_scope
        ),
    )

    try:
        # ``history`` carries the previous turns only; the current transcript
        # becomes the final user message inside the service. We append it to
        # the session's history *after* the LLM call returns.
        response = service.estimate_with_attachments(
            request,
            raw_attachments,
            project_metadata=session.metadata,
            history=session.history,
        )
    except AttachmentExtractionError as exc:
        log.info("session_estimate_attachment_rejected", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InputGuardrailViolation as exc:
        log.info(
            "session_estimate_blocked_by_input_guardrail",
            session_id=session_id,
            reason=exc.reason,
        )
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc
    except InstructorRetryException as exc:
        # Instructor exhausted its re-prompt budget — the LLM kept producing
        # an EstimationResult that fails a Pydantic validator (e.g. phases
        # do not sum to total_cost_eur). This is a client-fixable problem,
        # not an upstream failure: returning 422 lets the UI surface a
        # useful "try again with more detail" message instead of a 502.
        summary = _summarise_validation_failure(exc)
        log.info(
            "session_estimate_validation_exhausted",
            session_id=session_id,
            detail=summary,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "validation_exhausted",
                "message": (
                    f"The model could not produce a self-consistent "
                    f"estimation: {summary}. {_VALIDATION_FAILURE_HINT}"
                ),
                "validation_error": summary,
            },
        ) from exc
    except Exception as exc:
        log.error(
            "session_estimate_endpoint_error",
            session_id=session_id,
            error=str(exc)[:400],
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=502, detail="Upstream LLM call failed") from exc

    session.history.add("user", transcript)
    session.history.add("assistant", response.result.summary)

    # Step 4: refresh project_metadata from the just-completed turn. The
    # extractor is best-effort — a failure logs a warning and leaves the
    # previous metadata untouched (handled inside ``extract_project_metadata``).
    session.metadata = extract_project_metadata(
        current=session.metadata,
        transcript=transcript,
        summary=response.result.summary,
        llm_wrapper=get_llm_wrapper(),
    )

    return SessionEstimateResponse(
        estimation=response,
        project_metadata=session.metadata,
        completed_turns=session.history.completed_turns,
    )
