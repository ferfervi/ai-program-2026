"""Streamlit UI for the Session 9-11 grounded-estimate flow.

Drives the *latest* pipeline against a single endpoint:

    POST /v1/estimate/from-transcript   (auth: X-API-Key = ESTIMATE_API_KEY)

A raw meeting transcript goes in; a citation-backed, engineer-day ``Estimate``
comes out. Unlike the legacy ``streamlit_app.py`` (conversational sessions, euros
and weeks), this one renders the NEW schema:

* modules -> tasks, each task in engineer-days, each grounded task carrying
  per-line ``sources`` (chunk_id, document_id, verbatim evidence);
* a client-side **citation verification** panel rebuilt from those per-line
  flags (grounded / insufficient counts + the evidence behind every number).
  The endpoint already enforces "no dangling citations" internally (a corrective
  retry on fabricated ids), so this view surfaces *verifiability*, not policing.

Run it from the repo root::

    # The service must have a generous LLM timeout: gpt-5 with reasoning_effort=high
    # takes minutes and the default LLM_TIMEOUT=30 makes the endpoint 502.
    # Raise it in estimator/.env (LLM_TIMEOUT=600) and recreate the container:
    #   docker compose up -d --force-recreate estimator
    export ESTIMATE_API_KEY=...        # same value as estimator/.env
    streamlit run streamlit_app_s11.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


# The from-transcript router is mounted at the bare root (NOT under /api/v1).
API_BASE_URL = os.getenv("ESTIMATOR_BASE_URL", "http://localhost:8000")
FROM_TRANSCRIPT_PATH = "/v1/estimate/from-transcript"

# gpt-5 + reasoning_effort=high is slow; give the HTTP client plenty of room.
CLIENT_TIMEOUT_S = 900

# A transcript that grounds well against the shipped budgets corpus: mobile
# banking maps to BUD-2024-001 (mobile banking API, 480h) and BUD-2024-003
# (payments gateway), so retrieval finds strong analogs and most lines come
# back grounded with real evidence.
SAMPLE_TRANSCRIPT = """\
[Kickoff call — NeoBank mobile application]

Client (CTO): We're building a mobile banking application for the Spanish
market. We need OAuth 2.0 authentication with refresh-token rotation, and full
PSD2 strong customer authentication (SCA) for every payment initiation.

PM (Ana): Understood. You also mentioned an immutable transaction ledger?

Client (CTO): Yes — a double-entry transaction ledger as the source of truth for
balances, with an audit trail and nightly reconciliation jobs.

Backend Lead: We'll also need open banking connectors (AIS/PIS) plus token
revocation and introspection endpoints, and integration/security tests for the
auth flows.

Client (CTO): Timeline is tight — we target a regulated launch under Bank of
Spain supervision in about three months. Based on similar projects you've
delivered, how many engineer-days would the authentication and ledger work
usually take?

PM (Ana): This looks very close to the mobile banking API we delivered before.
We'll put together a grounded estimate with sources.
"""

CONFIDENCE_ICON = {"high": "🟢", "medium": "🟡", "low": "🟠", "insufficient": "🔴"}


def _set_page() -> None:
    st.set_page_config(
        page_title="Grounded Estimate — from transcript",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------


def _post_from_transcript(
    base_url: str, api_key: str, transcript: str, idempotency_key: str | None
) -> requests.Response:
    payload: Dict[str, Any] = {"transcript": transcript}
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    return requests.post(
        f"{base_url}{FROM_TRANSCRIPT_PATH}",
        json=payload,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        timeout=CLIENT_TIMEOUT_S,
    )


# ---------------------------------------------------------------------------
# Citation verification (rebuilt client-side from the Estimate)
# ---------------------------------------------------------------------------


def _citation_summary(estimate: Dict) -> Dict[str, int]:
    """Recompute the CitationReport aggregates from the per-line flags.

    The HTTP response carries the Estimate, not the server's CitationReport, but
    every task already exposes ``grounded`` + ``sources`` — enough to rebuild the
    grounded / insufficient counts and the number of verified citations. Dangling
    citations cannot be detected here (that needs the retrieved chunk ids, which
    the endpoint does not return); the endpoint already guarantees there are none.
    """
    total = grounded = insufficient = verified = 0
    for module in estimate.get("modules", []) or []:
        for task in module.get("tasks", []) or []:
            total += 1
            sources = task.get("sources", []) or []
            if task.get("grounded"):
                grounded += 1
                verified += len(sources)
            else:
                insufficient += 1
    return {
        "total_lines": total,
        "grounded_lines": grounded,
        "insufficient_lines": insufficient,
        "verified_citations": verified,
    }


def _render_header(estimate: Dict) -> None:
    confidence = estimate.get("confidence", "—")
    icon = CONFIDENCE_ICON.get(confidence, "")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total engineer-days", estimate.get("total_engineer_days") or "—")
    c2.metric("Duration (weeks)", estimate.get("duration_weeks") or "—")
    c3.metric("Confidence", f"{icon} {confidence}")

    if confidence == "insufficient":
        st.error(
            "The model judged the retrieved context insufficient to estimate "
            "responsibly.\n\n"
            + (estimate.get("insufficient_context_explanation") or "")
        )
    if estimate.get("reasoning"):
        with st.expander("Reasoning (how the estimate was derived)", expanded=False):
            st.write(estimate["reasoning"])


def _render_modules(estimate: Dict) -> None:
    """Render the modules -> tasks breakdown (the engineer-day 'phases')."""
    modules = estimate.get("modules", []) or []
    if not modules:
        st.caption("No modules in the estimate.")
        return

    st.subheader("Breakdown — modules → tasks (engineer-days)")
    for module in modules:
        m_days = sum(
            (t.get("engineer_days") or 0) for t in (module.get("tasks", []) or [])
        )
        with st.expander(f"📦 {module.get('name', '')}  ·  {m_days} engineer-days", expanded=True):
            if module.get("description"):
                st.caption(module["description"])
            for task in module.get("tasks", []) or []:
                grounded = task.get("grounded")
                badge = "✅ grounded" if grounded else "⚪ insufficient (no source data)"
                days = task.get("engineer_days")
                days_txt = f"**{days}d**" if days is not None else "—"
                st.markdown(f"- {days_txt} · {task.get('name', '')} — {badge}")
                if task.get("description"):
                    st.caption(f"    {task['description']}")
                for src in task.get("sources", []) or []:
                    st.markdown(
                        f"    > 📎 `chunk {src.get('chunk_id', '?')}` · "
                        f"`{src.get('document_id', '?')}` — *{src.get('evidence', '')}*"
                    )


def _render_citation_verification(estimate: Dict) -> None:
    s = _citation_summary(estimate)
    st.subheader("🔎 Citation verification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lines", s["total_lines"])
    c2.metric("Grounded", s["grounded_lines"])
    c3.metric("Insufficient", s["insufficient_lines"])
    c4.metric("Verified citations", s["verified_citations"])
    st.caption(
        "Every **grounded** line cites at least one retrieved chunk with verbatim "
        "evidence (shown under each task). **Insufficient** lines carry no invented "
        "hours. Dangling citations (ids never retrieved) are rejected server-side by "
        "the endpoint's corrective retry, so they are 0 by construction here."
    )


def _render_assumptions_and_sources(estimate: Dict) -> None:
    assumptions = estimate.get("assumptions", []) or []
    sources = estimate.get("sources", []) or []
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Assumptions** (scope without a historical analog)")
        if not assumptions:
            st.caption("None.")
        for a in assumptions:
            st.markdown(
                f"- *{a.get('impact', '')}* — {a.get('description', '')}  "
                f"\n  <small>{a.get('rationale', '')}</small>",
                unsafe_allow_html=True,
            )
    with col_b:
        st.markdown("**Estimate-global sources**")
        if not sources:
            st.caption("None.")
        for src in sources:
            st.markdown(
                f"- `chunk {src.get('source_id', '?')}` · {src.get('relevance', '')} — "
                f"{src.get('used_for', '')}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _set_page()
    st.title("📑 Grounded Estimate — from a transcript")
    st.caption(
        "Session 9-11 flow · `POST /v1/estimate/from-transcript` · "
        "engineer-day breakdown with verifiable per-line citations."
    )

    with st.sidebar:
        st.markdown("### Connection")
        base_url = st.text_input("Estimator base URL", value=API_BASE_URL)
        api_key = st.text_input(
            "ESTIMATE_API_KEY",
            value=os.getenv("ESTIMATE_API_KEY", ""),
            type="password",
            help="Header X-API-Key. Same value as estimator/.env.",
        )
        idempotency_key = st.text_input(
            "Idempotency key (optional)",
            value="",
            help="Reuse the same key to get the cached estimate without re-paying.",
        )
        st.divider()
        st.warning(
            "**Slow + timeout caveat.** Generation uses `gpt-5` with "
            "`reasoning_effort=high` — it takes minutes. The service default "
            "`LLM_TIMEOUT=30` will 502. Raise it in `estimator/.env` "
            "(`LLM_TIMEOUT=600`) and recreate the container:\n\n"
            "`docker compose up -d --force-recreate estimator`"
        )

    transcript = st.text_area(
        "Meeting transcript",
        value=SAMPLE_TRANSCRIPT,
        height=320,
        help="Free text. Minimum 100 characters (endpoint validation).",
    )
    submitted = st.button("Generate grounded estimate", type="primary", use_container_width=True)

    if not submitted:
        return

    if not api_key:
        st.error("Set ESTIMATE_API_KEY in the sidebar (or the environment).")
        return
    if len(transcript.strip()) < 100:
        st.warning("The transcript must be at least 100 characters.")
        return

    with st.spinner("Running reformulate → retrieve → assemble → generate → verify… (minutes)"):
        try:
            resp = _post_from_transcript(
                base_url.rstrip("/"), api_key, transcript.strip(), idempotency_key or None
            )
        except requests.RequestException as exc:
            st.error(f"Request failed (network / timeout): {exc}")
            return

    if resp.status_code == 401:
        st.error("401 Unauthorized — wrong or missing ESTIMATE_API_KEY.")
        return
    if resp.status_code == 422:
        st.warning(f"422 Validation error: {resp.text}")
        return
    if resp.status_code == 429:
        st.warning("429 Rate limited (10/min on this endpoint). Wait a minute and retry.")
        return
    if resp.status_code != 200:
        st.error(f"{resp.status_code} from the service: {resp.text}")
        return

    estimate = resp.json()
    if rid := resp.headers.get("X-Request-ID"):
        st.caption(f"X-Request-ID: `{rid}`")

    _render_header(estimate)
    st.divider()
    _render_citation_verification(estimate)
    st.divider()
    _render_modules(estimate)
    st.divider()
    _render_assumptions_and_sources(estimate)
    with st.expander("Raw Estimate JSON", expanded=False):
        st.json(estimate)


if __name__ == "__main__":
    main()
