import streamlit as st
import requests
import uuid
import sys
import re
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple


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
    </style>
""",
        unsafe_allow_html=True,
    )


def display_header():
    st.title("🤖 Project tasks estimator")
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


def display_sidebar():
    with st.sidebar:
        st.header("Session Info")
        st.write(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 New Session"):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.markdown("### How it works:")
        st.markdown("""
    1. Type your transcription of the meeting task definition
    2. The assistant provides an estimation based on the transcription and the company guidelines
    """)
        if st.button("Test Connection to AI Server"):
            try:
                res = requests.get("http://localhost:8000/health") # O la ruta raíz de tu API
                st.write(f"Respuesta del backend: {res.json()}")
            except Exception as e:
                st.error(f"Fallo total: {e}")


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
                            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
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
                        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                        with st.expander("📊 Debug Info"):
                            display_metadata(event)
                        return

            if assistant_text:
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                with st.expander("📊 Debug Info"):
                    display_metadata({
                        "model": os.getenv("LLM_MODEL", ""),
                        "provider": os.getenv("LLM_PROVIDER", ""),
                        "timestamp": str(datetime.utcnow()),
                    })
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
    display_sidebar()
    display_chat_history()
    handle_user_interaction()


if __name__ == "__main__":
    main()
