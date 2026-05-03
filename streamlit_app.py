import streamlit as st
import requests
import uuid
import sys
import re
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.services.llm_service import build_system_prompt
from app.context.examples import ESTIMATION_EXAMPLES


API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/estimate")
STREAM_API_URL = f"{API_URL}/stream"


# -----------------
# Helper functions
# -----------------

def set_page():
    st.set_page_config(
        page_title="Project tasks estimator",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
    )


def inject_css():
    st.markdown(
        """
    <style>
        body { background-color: #f5f5f5; }
        .main { padding: 2rem; }
        section[data-testid="stSidebar"] button {
            padding: 0.35rem 0.75rem !important;
            font-size: 0.88rem !important;
            min-height: 32px !important;
            line-height: 1.2 !important;
        }
        section[data-testid="stSidebar"] .stButton > button {
            width: auto !important;
        }
    </style>
""",
        unsafe_allow_html=True,
    )


def display_header():
    st.title("🚀 Project task estimator")
    st.markdown(
        f"""
        Estimate the time and resources needed for your project tasks.

        This estimator uses historical data and company guidelines to provide accurate estimates.
        The API runs is accessible at **{API_URL}** (configurable via `API_URL` env var).
        """ 
    )


def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_call_metrics" not in st.session_state:
        st.session_state.last_call_metrics = {}


def display_sidebar():
    with st.sidebar:
        st.markdown("### How it works:")
        st.caption("""1. Type your transcription of the meeting task definition
2. The assistant provides an estimation based on the transcription and the company guidelines""")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            st.write(" ")
            st.write(" ")

        if st.button("Test Connection to AI Server"):
            try:
                res = requests.get("http://localhost:8000/health") # O la ruta raíz de tu API
                st.write(f"Respuesta del backend: {res.json()}")
            except Exception as e:
                st.error(f"Fallo total: {e}")

        st.divider()
        st.markdown("### 📊 Last call metrics")
        if st.session_state.last_call_metrics:
            metrics = st.session_state.last_call_metrics
            st.write("**Model:**", metrics.get("model", ""))
            st.write("**Input tokens:**", metrics.get("input_tokens", ""))
            st.write("**Output tokens:**", metrics.get("output_tokens", ""))
            st.write("**Response time:**", f"{metrics.get('response_time_ms', 0)} ms")
            st.write("\n_Updated after the latest estimation call._")
        else:
            st.write("🤷‍♂️ No hay métricas de llamada aún.")
            st.write("Envía una estimación para ver modelo, tokens y tiempo de respuesta aquí.")

        st.divider()
        st.markdown("### 🧠 Active system prompt")
        st.text_area("System prompt", build_system_prompt(), height=200, disabled=True)

        st.divider()
        st.markdown("### Static injected context")
        for index, example in enumerate(ESTIMATION_EXAMPLES, start=1):
            with st.expander(f"Ejemplo {index}"):
                st.markdown(f"**Resumen de la reunión:** {example.get('meeting_summary', '').strip()}")
                st.markdown("**Estimación:**")
                st.code(example.get('estimation', '').strip())

        st.divider()
        st.header("Session Info")
        st.write(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
        if st.button("🔄 New Session"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()


def display_chat_history():
    st.markdown("---")
    if st.session_state.messages:
        st.markdown("### Conversation")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.info("👋 No messages yet. Start by asking a question below!")



def display_metadata(data: Dict):
    model = data.get("model", "")
    provider = data.get("provider", "")
    timestamp = data.get("timestamp", "")
    token_usage = data.get("token_usage", {}) or {}

    st.write("**Model:**", model)
    st.write("**Provider:**", provider)
    st.write("**Timestamp:**", str(timestamp))

    if token_usage:
        st.write("**Input tokens:**", token_usage.get("input_tokens"))
        st.write("**Output tokens:**", token_usage.get("output_tokens"))
        st.write("**Total tokens:**", token_usage.get("total_tokens"))


def send_query(api_url: str, payload: Dict, timeout: int = 30) -> requests.Response:
    return requests.post(api_url, json=payload, timeout=timeout)


def send_stream_query(api_url: str, payload: Dict, timeout: tuple[int, Optional[float]] = (5, None)) -> requests.Response:
    headers = {"Accept": "text/event-stream"}
    return requests.post(api_url, json=payload, timeout=timeout, stream=True, headers=headers)


def parse_sse_event(lines: List[str]) -> Optional[Dict]:
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


def handle_user_interaction():
    user_input = st.chat_input("Provide a transcription of the meeting task definition...")
    if not user_input or not user_input.strip():
        return

    # Display user message and save
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call backend
    with st.spinner("🔍 Providing estimation..."):
        try:
            start_time = datetime.utcnow()
            payload = {"transcription": user_input, "thread_id": st.session_state.thread_id}
            resp = send_stream_query(STREAM_API_URL, payload)

            if resp.status_code != 200:
                st.error(f"❌ Server error: {resp.status_code}")
                st.write(resp.text)
                return

            assistant_text = ""
            event_lines: List[str] = []
            with st.chat_message("assistant"):
                assistant_placeholder = st.empty()
                status_placeholder = st.empty()
                status_placeholder.info("🟢 Generating estimation... Please wait.")
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if raw_line is None:
                        continue

                    line = raw_line.strip()
                    if line == "":
                        event = parse_sse_event(event_lines)
                        event_lines = []
                        if not event:
                            continue

                        event_type = event.get("type")
                        if event_type == "delta":
                            assistant_text += event.get("text", "")
                            assistant_placeholder.markdown(assistant_text)
                        elif event_type == "done":
                            assistant_text = event.get("estimation", assistant_text)
                            assistant_placeholder.markdown(assistant_text)
                            status_placeholder.success("✅ Estimation complete.")
                            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                            st.session_state.last_call_metrics = {
                                "model": event.get("model", ""),
                                "input_tokens": event.get("token_usage", {}).get("input_tokens", 0),
                                "output_tokens": event.get("token_usage", {}).get("output_tokens", 0),
                                "response_time_ms": response_time_ms,
                            }
                            with st.expander("📊 Debug Info"):
                                display_metadata(event)
                            return
                        continue

                    event_lines.append(line)

                if event_lines:
                    event = parse_sse_event(event_lines)
                    if event and event.get("type") == "done":
                        assistant_text = event.get("estimation", assistant_text)
                        assistant_placeholder.markdown(assistant_text)
                        status_placeholder.success("✅ Estimation complete.")
                        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                        response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                        st.session_state.last_call_metrics = {
                            "model": event.get("model", ""),
                            "input_tokens": event.get("token_usage", {}).get("input_tokens", 0),
                            "output_tokens": event.get("token_usage", {}).get("output_tokens", 0),
                            "response_time_ms": response_time_ms,
                        }
                        with st.expander("📊 Debug Info"):
                            display_metadata(event)
                        return

            if assistant_text:
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            else:
                st.error("❌ No se recibieron datos de streaming.")

        except requests.exceptions.ConnectionError:
            st.error(
                """
            ❌ **Cannot connect to server**

            Make sure the backend is running:
            ```bash
            uv run uvicorn app.main:app --reload
            ```
            """
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The server took too long to respond.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# -----------------
# Main
# -----------------

def main():
    set_page()
    inject_css()
    display_header()
    init_session()
    display_chat_history()
    handle_user_interaction()
    display_sidebar()


if __name__ == "__main__":
    main()
