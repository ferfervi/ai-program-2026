import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
import streamlit as st

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import EstimationRequest, ProjectType, DetailLevel, OutputFormat

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/estimate")
STREAM_API_URL = f"{API_URL}/stream"

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
# Page setup
# ---------------------------------------------------------------------------

def _set_page():
    st.set_page_config(
        page_title="Project Estimator",
        page_icon="📋",
        layout="centered",
        initial_sidebar_state="auto",
    )


def _init_session():
    if "last_call_metrics" not in st.session_state:
        st.session_state.last_call_metrics = {}
    if "last_result" not in st.session_state:
        st.session_state.last_result = ""
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = ""
    if "last_system_prompt" not in st.session_state:
        st.session_state.last_system_prompt = ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _display_sidebar():
    with st.sidebar:
        if st.button("Test connection"):
            try:
                res = requests.get("http://localhost:8000/health", timeout=5)
                st.success(f"Backend OK — {res.json()}")
            except Exception as e:
                st.error(f"Cannot connect: {e}")

        st.divider()
        st.markdown("### Last call metrics")
        if st.session_state.last_call_metrics:
            m = st.session_state.last_call_metrics
            st.write("**Model:**", m.get("model", "—"))
            st.write("**Provider:**", m.get("provider", "—"))
            st.write("**Input tokens:**", m.get("input_tokens", 0))
            st.write("**Output tokens:**", m.get("output_tokens", 0))
            st.write("**Cost:**", f"${m.get('cost_usd', 0.0):.6f}")
            st.write("**Latency:**", f"{m.get('latency_ms', 0)} ms")
            st.write("**Cache hit:**", m.get("cache_hit", False))
        else:
            st.caption("No metrics yet — submit a request first.")

        st.divider()
        st.markdown("### Last system prompt")
        if st.session_state.last_system_prompt:
            st.text_area("System prompt", st.session_state.last_system_prompt, height=200, disabled=True)
        else:
            st.caption("Submit a request to see the rendered system prompt here.")

        st.divider()
        st.markdown("### Last user prompt")
        if st.session_state.last_user_prompt:
            st.text_area("User prompt", st.session_state.last_user_prompt, height=120, disabled=True)
        else:
            st.caption("Submit a request to see the rendered user prompt here.")

        st.divider()


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

def _parse_sse_event(lines: List[str]) -> Optional[Dict]:
    event_type = None
    data_lines: List[str] = []
    for line in lines:
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if not data_lines:
        return None
    try:
        data = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
    if event_type:
        data["type"] = event_type
    return data


def _stream_estimation(payload: Dict) -> None:
    headers = {"Accept": "text/event-stream"}
    result_placeholder = st.empty()
    status_placeholder = st.empty()

    try:
        start = datetime.now(timezone.utc)
        resp = requests.post(
            STREAM_API_URL, json=payload, stream=True,
            headers=headers, timeout=(5, None),
        )
        if resp.status_code != 200:
            st.error(f"Server error {resp.status_code}: {resp.text}")
            return

        accumulated = ""
        event_lines: List[str] = []

        with st.spinner("Generating estimation…"):
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if line == "":
                    event = _parse_sse_event(event_lines)
                    event_lines = []
                    if not event:
                        continue
                    if event.get("type") == "delta":
                        accumulated += event.get("text", "")
                        result_placeholder.markdown(accumulated)
                    elif event.get("type") == "done":
                        accumulated = event.get("estimation", accumulated)
                        result_placeholder.markdown(accumulated)
                        status_placeholder.success("Estimation complete")
                        latency_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
                        st.session_state.last_result = accumulated
                        st.session_state.last_call_metrics = {
                            "model": event.get("model", ""),
                            "provider": event.get("provider", ""),
                            "input_tokens": event.get("token_usage", {}).get("input_tokens", 0),
                            "output_tokens": event.get("token_usage", {}).get("output_tokens", 0),
                            "cost_usd": event.get("token_usage", {}).get("cost_usd", 0.0),
                            "latency_ms": event.get("latency_ms", latency_ms),
                            "cache_hit": event.get("cache_hit", False),
                        }
                        return
                    continue
                event_lines.append(line)

        if not accumulated:
            st.error("No data received from stream.")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is `make serve` running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
    except Exception as e:
        st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Blocking (non-streaming) call
# ---------------------------------------------------------------------------

def _render_estimation_result(result: Dict) -> str:
    """Render a structured EstimationResult into markdown for display + history."""
    summary = result.get("summary", "")
    confidence = result.get("confidence_pct", 0)
    phases = result.get("phases", []) or []
    total_weeks = result.get("total_duration_weeks", 0)
    total_cost = result.get("total_cost_eur", 0)

    lines = [
        f"### Summary",
        summary,
        "",
        f"**Confidence:** {confidence}%",
        "",
        "### Phases",
        "| # | Phase | Weeks | Cost (EUR) | Summary |",
        "|---|---|---:|---:|---|",
    ]
    for i, p in enumerate(phases, start=1):
        phase_summary = (p.get("summary", "") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {p.get('name','')} | {p.get('duration_weeks',0)} "
            f"| {p.get('cost_eur',0):,} | {phase_summary} |"
        )
    lines += [
        "",
        f"**Total duration:** {total_weeks} weeks  ",
        f"**Total cost:** €{total_cost:,}",
    ]
    return "\n".join(lines)


def _blocking_estimation(payload: Dict) -> None:
    try:
        with st.spinner("Generating estimation…"):
            resp = requests.post(API_URL, json=payload, timeout=120)
        if resp.status_code != 200:
            st.error(f"Server error {resp.status_code}: {resp.text}")
            return
        data = resp.json()
        result = data.get("result") or {}
        if not result:
            st.error("Response did not include a `result` object.")
            st.json(data)
            return
        rendered = _render_estimation_result(result)
        st.markdown(rendered)
        st.success("Estimation complete")
        usage = data.get("usage") or {}
        st.session_state.last_result = rendered
        st.session_state.last_call_metrics = {
            "model": data.get("model", ""),
            "provider": data.get("provider", ""),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": data.get("cost_usd", 0.0),
            "latency_ms": data.get("latency_ms", 0),
            "cache_hit": data.get("cached", False),
        }
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is `make serve` running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
    except Exception as e:
        st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _set_page()
    _init_session()

    st.title("Project Estimator")
    st.caption(f"API: `{API_URL}` — configurable via `API_URL` env var")

    with st.form("estimation_form"):
        description = st.text_area(
            "Project description",
            placeholder="Paste or type the meeting transcription or project requirements here…",
            height=160,
        )

        col_left, col_right = st.columns(2)
        with col_left:
            project_type_label = st.selectbox(
                "Project type",
                options=list(PROJECT_TYPE_LABELS.keys()),
            )
        with col_right:
            output_format_label = st.selectbox(
                "Output format",
                options=list(OUTPUT_FORMAT_LABELS.keys()),
            )

        detail_level_label = st.radio(
            "Detail level",
            options=DETAIL_LEVEL_LABELS,
            index=0,
            horizontal=True,
        )

        streaming = st.toggle("Streaming", value=False)

        submitted = st.form_submit_button(
            "Generate estimation", type="primary", use_container_width=True
        )

    if submitted:
        if not description or len(description.strip()) < 20:
            st.warning("Description must be at least 20 characters.")
        else:
            payload = {
                "description": description.strip(),
                "project_type": PROJECT_TYPE_LABELS[project_type_label],
                "detail_level": DETAIL_LEVEL_VALUES[DETAIL_LEVEL_LABELS.index(detail_level_label)],
                "output_format": OUTPUT_FORMAT_LABELS[output_format_label],
            }
            system_prompt, user_prompt = render_estimation_prompt(
                EstimationRequest(
                    description=payload["description"],
                    project_type=ProjectType(payload["project_type"]),
                    detail_level=DetailLevel(payload["detail_level"]),
                    output_format=OutputFormat(payload["output_format"]),
                )
            )
            st.session_state.last_system_prompt = system_prompt
            st.session_state.last_user_prompt = user_prompt
            st.markdown("---")
            if streaming:
                _stream_estimation(payload)
            else:
                _blocking_estimation(payload)
    elif st.session_state.last_result:
        st.markdown("---")
        st.markdown(st.session_state.last_result)

    _display_sidebar()


if __name__ == "__main__":
    main()
