"""Streamlit UI for the conversational sessions endpoint.

Drives the modern multi-turn flow against:
    POST /api/v1/sessions
    POST /api/v1/sessions/{session_id}/estimate  (multipart)
    GET  /api/v1/sessions/{session_id}

Key UX choices:
- A session is created lazily on first page load and its ``session_id`` is
  kept in ``st.session_state`` so reruns stay on the same conversation.
- The sidebar shows the *current* ``project_metadata`` (memoria) and the
  raw windowed ``history`` (historial) side by side — the visual separation
  reinforces that the two structures live independently.
- A "Nueva conversación" button hits ``POST /sessions`` again and resets
  every Streamlit state key, giving the user a clean slate.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

PROJECT_TYPE_LABELS: Dict[str, str] = {
    "Web SaaS": "web_saas",
    "Mobile App": "mobile_app",
    "Internal Tool": "internal_tool",
    "Data Pipeline": "data_pipeline",
}

OUTPUT_FORMAT_LABELS: Dict[str, str] = {
    "Phases Table": "phases_table",
    "Line Items": "line_items",
    "Narrative": "narrative",
}

DETAIL_LEVEL_LABELS = ["Summary", "Medium", "Detailed"]
DETAIL_LEVEL_VALUES = ["summary", "medium", "detailed"]


# ---------------------------------------------------------------------------
# Page setup + session bootstrapping
# ---------------------------------------------------------------------------


def _set_page() -> None:
    st.set_page_config(
        page_title="Project Estimator — Sessions",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _create_session() -> str:
    resp = requests.post(f"{API_BASE_URL}/sessions", timeout=10)
    resp.raise_for_status()
    return resp.json()["session_id"]


def _ensure_session() -> None:
    """Create a session lazily on first run; do nothing afterwards."""
    if "session_id" in st.session_state:
        return
    try:
        st.session_state.session_id = _create_session()
    except Exception as exc:
        st.error(
            f"Cannot create a session against `{API_BASE_URL}`.\n\n"
            f"Is the backend running? Error: {exc}"
        )
        st.stop()
    st.session_state.turns = []
    st.session_state.project_metadata = {}
    st.session_state.history = []
    st.session_state.completed_turns = 0
    st.session_state.max_turns = 6
    st.session_state.last_call_metrics = {}
    # Monotonic counter used to derive widget keys for the transcript + file
    # uploader. Incrementing it after a successful turn forces Streamlit to
    # instantiate brand-new widgets on the next run, which is the only
    # reliable way to clear an ``st.file_uploader`` (popping its key from
    # session_state does not visually reset the displayed files).
    st.session_state.form_version = 0


def _reset_session() -> None:
    """Drop all state and open a brand new session."""
    for key in (
        "session_id",
        "turns",
        "project_metadata",
        "history",
        "completed_turns",
        "max_turns",
        "last_call_metrics",
    ):
        st.session_state.pop(key, None)
    _ensure_session()


def _refresh_from_server() -> None:
    """Pull the authoritative session state from the backend.

    Called on every rerun so the sidebar (memoria + historial) stays in
    sync with whatever the extractor did after the previous turn.
    """
    if "session_id" not in st.session_state:
        return
    try:
        resp = requests.get(
            f"{API_BASE_URL}/sessions/{st.session_state.session_id}", timeout=10
        )
    except requests.RequestException:
        return
    if resp.status_code != 200:
        return
    state = resp.json()
    st.session_state.project_metadata = state.get("project_metadata", {})
    st.session_state.history = state.get("history", [])
    st.session_state.completed_turns = state.get("completed_turns", 0)
    st.session_state.max_turns = state.get("max_turns", st.session_state.max_turns)


# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------


def _post_estimate(
    transcript: str,
    attachments: List[Any],
    project_type: str,
    detail_level: str,
    output_format: str,
) -> requests.Response:
    files = None
    if attachments:
        files = [
            (
                "attachments",
                (
                    upload.name,
                    upload.getvalue(),
                    upload.type or "application/octet-stream",
                ),
            )
            for upload in attachments
        ]
    data = {
        "transcript": transcript,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    return requests.post(
        f"{API_BASE_URL}/sessions/{st.session_state.session_id}/estimate",
        data=data,
        files=files,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_estimation_result(result: Dict, output_format: str = "phases_table") -> str:
    """Same canonical renderer as the stateless form UI — one schema, three
    presentations chosen by ``output_format``."""
    summary = result.get("summary", "")
    confidence = result.get("confidence_pct", 0)
    phases = result.get("phases", []) or []
    total_weeks = result.get("total_duration_weeks", 0)
    total_cost = result.get("total_cost_eur", 0)

    header = [
        "### Summary",
        summary,
        "",
        f"**Confidence:** {confidence}%",
        "",
    ]

    if output_format == "line_items":
        body = ["### Phases"]
        for i, p in enumerate(phases, start=1):
            body.append(
                f"{i}. **{p.get('name','')}** — {p.get('duration_weeks',0)} week(s), "
                f"€{p.get('cost_eur',0):,}. {p.get('summary','')}"
            )
        footer = ["", f"**Totals:** {total_weeks} weeks · €{total_cost:,}"]
        return "\n".join(header + body + footer)

    if output_format == "narrative":
        paragraphs: List[str] = []
        for p in phases:
            paragraphs.append(
                f"During **{p.get('name','')}** (about {p.get('duration_weeks',0)} week(s), "
                f"~€{p.get('cost_eur',0):,}), {p.get('summary','')}"
            )
        footer = [
            "",
            f"Overall, the project runs for **{total_weeks} weeks** at a total of "
            f"**€{total_cost:,}**.",
        ]
        return "\n".join(header + paragraphs + footer)

    body = [
        "### Phases",
        "| # | Phase | Weeks | Cost (EUR) | Summary |",
        "|---|---|---:|---:|---|",
    ]
    for i, p in enumerate(phases, start=1):
        phase_summary = (p.get("summary", "") or "").replace("|", "\\|").replace("\n", " ")
        body.append(
            f"| {i} | {p.get('name','')} | {p.get('duration_weeks',0)} "
            f"| {p.get('cost_eur',0):,} | {phase_summary} |"
        )
    footer = [
        "",
        f"**Total duration:** {total_weeks} weeks  ",
        f"**Total cost:** €{total_cost:,}",
    ]
    return "\n".join(header + body + footer)


def _has_any_metadata(md: Dict) -> bool:
    if not md:
        return False
    return any(
        [
            md.get("project_name"),
            md.get("assumed_team_size"),
            md.get("mentioned_technologies"),
            md.get("agreed_scope"),
            md.get("explicit_constraints"),
            md.get("rejected_options"),
        ]
    )


# ---------------------------------------------------------------------------
# Sidebar — memoria + historial side by side
# ---------------------------------------------------------------------------


def _render_metadata_panel(md: Dict) -> None:
    if not _has_any_metadata(md):
        st.caption("Empty — facts will appear here as the conversation progresses.")
        return
    if md.get("project_name"):
        st.write(f"**Project name:** {md['project_name']}")
    if md.get("assumed_team_size"):
        st.write(f"**Team size:** {md['assumed_team_size']}")
    if md.get("mentioned_technologies"):
        st.write(f"**Technologies:** {', '.join(md['mentioned_technologies'])}")
    if md.get("agreed_scope"):
        st.write(f"**Agreed scope:** {md['agreed_scope']}")
    if md.get("explicit_constraints"):
        st.write("**Explicit constraints:**")
        for item in md["explicit_constraints"]:
            st.write(f"- {item}")
    if md.get("rejected_options"):
        st.write("**Rejected options:**")
        for item in md["rejected_options"]:
            st.write(f"- {item}")


def _render_history_panel(history: List[Dict]) -> None:
    if not history:
        st.caption("No messages yet.")
        return
    # Newest first so the user reads top-down without scrolling past stale turns.
    for entry in reversed(history):
        role = entry.get("role", "?")
        icon = "🧑" if role == "user" else "🤖"
        content = entry.get("content", "")
        snippet = content if len(content) <= 600 else content[:600] + "…"
        st.markdown(f"**{icon} {role.capitalize()}:** {snippet}")


# ---------------------------------------------------------------------------
# Session info dialog
# ---------------------------------------------------------------------------


@st.dialog("Session info", width="large")
def _session_info_dialog() -> None:
    """Modal pop-up with the full GET /sessions/{id} payload.

    Fetches the authoritative state on open so what the user sees matches
    what the backend actually holds (not the last cached copy in
    ``st.session_state``).
    """
    session_id = st.session_state.get("session_id")
    if not session_id:
        st.warning("No active session.")
        return

    try:
        resp = requests.get(f"{API_BASE_URL}/sessions/{session_id}", timeout=10)
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return

    if resp.status_code != 200:
        st.error(f"Backend returned {resp.status_code}: {resp.text}")
        return

    state = resp.json()

    st.markdown(f"**Session id:** `{state.get('session_id', session_id)}`")
    st.markdown(
        f"**Turns:** {state.get('completed_turns', 0)} / "
        f"{state.get('max_turns', 0)} (sliding-window capacity)"
    )

    st.divider()
    st.subheader("🧠 Project metadata (memoria)")
    _render_metadata_panel(state.get("project_metadata") or {})

    st.divider()
    st.subheader("💬 Conversation history (historial)")
    history = state.get("history") or []
    if not history:
        st.caption("No messages yet.")
    else:
        st.caption(f"{len(history)} message(s) — newest first")
        _render_history_panel(history)

    st.divider()
    with st.expander("Raw JSON", expanded=False):
        st.json(state)


def _render_sidebar() -> None:
    with st.sidebar:
        if st.button(
            "🔄 Nueva conversación",
            use_container_width=True,
            type="primary",
            help="Calls POST /sessions and resets every Streamlit state key.",
        ):
            _reset_session()
            st.rerun()

        st.divider()
        st.markdown("### Session")
        st.code(st.session_state.session_id, language=None)
        completed = st.session_state.get("completed_turns", 0)
        max_t = st.session_state.get("max_turns", 6)
        st.write(f"**Turns:** {completed} / {max_t}")
        col_refresh, col_info = st.columns(2)
        with col_refresh:
            if st.button("Refresh", use_container_width=True):
                _refresh_from_server()
                st.rerun()
        with col_info:
            if st.button(
                "📋 Info",
                use_container_width=True,
                help="Open the full session snapshot (history + project_metadata) in a dialog.",
            ):
                _session_info_dialog()

        st.divider()
        st.markdown("### 🧠 Project metadata (memoria)")
        st.caption(
            "Distilled facts. Survives history truncation; injected into the "
            "system prompt every turn."
        )
        _render_metadata_panel(st.session_state.get("project_metadata") or {})

        st.divider()
        with st.expander("💬 Conversation history (historial)", expanded=False):
            st.caption(
                "Raw windowed log of (user, assistant) pairs sent to the LLM. "
                f"Capacity: {max_t} pairs (sliding window)."
            )
            _render_history_panel(st.session_state.get("history") or [])

        st.divider()
        st.markdown("### Last call metrics")
        m = st.session_state.get("last_call_metrics") or {}
        if not m:
            st.caption("No turn submitted yet.")
        else:
            st.write(f"Model: {m.get('model', '—')}")
            st.write(f"Provider: {m.get('provider', '—')}")
            st.write(f"Cost: ${m.get('cost_usd', 0.0):.6f}")
            st.write(f"Latency: {m.get('latency_ms', 0)} ms")
            st.write(f"Cached: {m.get('cached', False)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _set_page()
    _ensure_session()
    _refresh_from_server()

    st.title("Project Estimator — Conversational")
    st.caption(
        f"API base: `{API_BASE_URL}` · session `{st.session_state.session_id}`"
    )

    # Keys derived from ``form_version`` so the next turn renders fresh widgets.
    form_v = st.session_state.get("form_version", 0)
    with st.form("session_turn_form", clear_on_submit=False):
        transcript = st.text_area(
            "Transcription (new turn)",
            placeholder=(
                "Paste the latest meeting note or refinement here. The model "
                "will incorporate previous turns + project_metadata."
            ),
            height=160,
            key=f"turn_transcript_{form_v}",
        )

        attachments = st.file_uploader(
            "Attachments (optional)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="PDF / DOCX / TXT / MD — extracted locally and appended to the transcript.",
            key=f"turn_attachments_{form_v}",
        )

        col_left, col_right = st.columns(2)
        with col_left:
            project_type_label = st.selectbox(
                "Project type", options=list(PROJECT_TYPE_LABELS.keys())
            )
        with col_right:
            output_format_label = st.selectbox(
                "Output format", options=list(OUTPUT_FORMAT_LABELS.keys())
            )

        detail_level_label = st.radio(
            "Detail level",
            options=DETAIL_LEVEL_LABELS,
            index=0,
            horizontal=True,
        )

        submitted = st.form_submit_button(
            "Send turn", type="primary", use_container_width=True
        )

    if submitted:
        if not transcript or len(transcript.strip()) < 20:
            st.warning("The transcript must be at least 20 characters.")
        else:
            project_type = PROJECT_TYPE_LABELS[project_type_label]
            detail_level = DETAIL_LEVEL_VALUES[
                DETAIL_LEVEL_LABELS.index(detail_level_label)
            ]
            output_format = OUTPUT_FORMAT_LABELS[output_format_label]

            with st.spinner("Sending turn…"):
                try:
                    resp = _post_estimate(
                        transcript.strip(),
                        attachments or [],
                        project_type,
                        detail_level,
                        output_format,
                    )
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                else:
                    if resp.status_code == 422:
                        # Validator exhausted (e.g. phases don't sum to the
                        # total). The route returns a structured payload with
                        # a user-friendly message and the validator's own
                        # complaint — surface them as a warning, not an
                        # opaque "server error".
                        detail = {}
                        try:
                            detail = resp.json().get("detail", {}) or {}
                        except ValueError:
                            pass
                        if isinstance(detail, dict) and detail.get("message"):
                            st.warning(detail["message"])
                            if detail.get("validation_error"):
                                st.caption(
                                    f"Validator: {detail['validation_error']}"
                                )
                        else:
                            st.error(
                                f"Validation failed (422). Response: {resp.text}"
                            )
                    elif resp.status_code != 200:
                        st.error(f"Server error {resp.status_code}: {resp.text}")
                    else:
                        body = resp.json()
                        estimation = body.get("estimation", {})
                        result = estimation.get("result") or {}

                        rendered = _render_estimation_result(result, output_format)
                        st.session_state.turns.append(
                            {
                                "transcript": transcript.strip(),
                                "rendered": rendered,
                                "attachments": [
                                    f.name for f in (attachments or [])
                                ],
                                "extractions": estimation.get("attachments", []),
                                "output_format": output_format,
                            }
                        )
                        usage = estimation.get("usage") or {}
                        st.session_state.last_call_metrics = {
                            "model": estimation.get("model", ""),
                            "provider": estimation.get("provider", ""),
                            "cost_usd": estimation.get("cost_usd", 0.0),
                            "latency_ms": estimation.get("latency_ms", 0),
                            "cached": estimation.get("cached", False),
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        }
                        st.session_state.project_metadata = body.get(
                            "project_metadata", {}
                        )
                        st.session_state.completed_turns = body.get(
                            "completed_turns", 0
                        )
                        # Bump the form version so the transcript + uploader
                        # render as brand-new widgets on the next run (the
                        # only reliable way to reset st.file_uploader). The
                        # selectors keep their values — they are not keyed
                        # off form_version on purpose.
                        old_v = st.session_state.get("form_version", 0)
                        st.session_state.pop(f"turn_transcript_{old_v}", None)
                        st.session_state.pop(f"turn_attachments_{old_v}", None)
                        st.session_state.form_version = old_v + 1
                        # Re-run so the sidebar reflects the just-updated state
                        # (and so a fresh GET /sessions/{id} populates the
                        # history panel with the new pair).
                        st.rerun()

    # Render the conversation so the user sees the full session at a glance.
    # Newest turn on top: when a long session scrolls, the latest result is
    # always visible without paging through stale ones.
    if st.session_state.get("turns"):
        st.markdown("---")
        st.subheader("Session turns")
        total = len(st.session_state.turns)
        for offset, turn in enumerate(reversed(st.session_state.turns)):
            turn_number = total - offset
            is_latest = offset == 0
            with st.container():
                heading = f"#### Turn {turn_number}"
                if is_latest:
                    heading += "  · latest"
                st.markdown(heading)
                if turn["attachments"]:
                    st.caption(
                        f"📎 Attachments: {', '.join(turn['attachments'])}"
                    )
                with st.expander("Transcript", expanded=False):
                    st.write(turn["transcript"])
                extractions = turn.get("extractions") or []
                if extractions:
                    with st.expander(
                        f"📄 Extracted from attachments ({len(extractions)} file"
                        f"{'s' if len(extractions) != 1 else ''}) — what the LLM saw",
                        expanded=False,
                    ):
                        for att in extractions:
                            st.markdown(
                                f"**{att.get('filename','')}** — "
                                f"{att.get('chars', 0):,} chars · "
                                f"{att.get('bytes', 0):,} bytes"
                            )
                            text = att.get("text", "")
                            if text:
                                st.text(text)
                            else:
                                st.caption(
                                    "Empty extraction (scanned PDF without an "
                                    "OCR layer, or unsupported content)."
                                )
                            st.divider()
                st.markdown(turn["rendered"])
                st.markdown("---")

    _render_sidebar()


if __name__ == "__main__":
    main()
