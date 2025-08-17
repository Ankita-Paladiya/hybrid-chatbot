import os
import streamlit as st
from typing import List, Dict, Tuple

# Import your chatbot logic from code.py
from code import (
    hybrid_chatbot,
    llm_chatbot,
    traditional_chatbot,
    chatbot_dashboard,
    user_memory,
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Hybrid Chatbot",
    page_icon="🤖",
    layout="centered",
)

# -----------------------------
# Helpers
# -----------------------------
LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
}

BOT_CHOICES = {
    "Hybrid": "hybrid",
    "Traditional (rules)": "traditional",
    "LLM-only": "llm",
}

def ensure_openai_key():
    """Resolve the OpenAI API key from Streamlit secrets or a sidebar input and set env var."""
    key_from_secrets = st.secrets.get("OPENAI_API_KEY", None)
    if key_from_secrets:
        os.environ["OPENAI_API_KEY"] = key_from_secrets
        return True

    key_input = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Used by code.py for GPT calls.",
    )
    if key_input:
        os.environ["OPENAI_API_KEY"] = key_input
        return True
    return False

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # type: List[Dict[str, str]]

if "confidences" not in st.session_state:
    st.session_state.confidences = []  # type: List[float]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")
key_ok = ensure_openai_key()

bot_mode_label = st.sidebar.selectbox("Model", list(BOT_CHOICES.keys()), index=0)
bot_mode = BOT_CHOICES[bot_mode_label]

language_label = st.sidebar.selectbox("Response language", list(LANG_MAP.keys()), index=0)
language_code = LANG_MAP[language_label]

show_conf = st.sidebar.checkbox("Show confidence", value=True)
show_metrics = st.sidebar.checkbox("Show dashboard metrics", value=False)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.session_state.confidences = []
    # Also clear global memory from code.py (session-level)
    user_memory.clear()
    st.sidebar.success("Chat and memory cleared.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` for seamless deploys."
)

# -----------------------------
# Main UI
# -----------------------------
st.title("🤖 Hybrid Chatbot")
st.write(
    "Ask anything! Choose Hybrid for best overall performance, or switch to Rules/LLM to compare."
)

if not key_ok:
    st.info("Add your OpenAI API key in the sidebar to enable LLM responses.")

# Display existing chat
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"]) 
        if show_conf and msg["role"] == "assistant" and i < len(st.session_state.confidences):
            st.caption(f"Confidence: {st.session_state.confidences[i]:.2f}%")

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Show the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        if bot_mode == "hybrid":
            # Hybrid supports multilingual via target_language
            response, conf = hybrid_chatbot(prompt, target_language=language_code)
        elif bot_mode == "traditional":
            response, conf = traditional_chatbot(prompt)
        else:
            response, conf = llm_chatbot(prompt)

        st.markdown(response)
        if show_conf:
            st.caption(f"Confidence: {conf:.2f}%")

    # Persist
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.confidences.append(conf)

# Optional dashboard from code.py (uses global user_memory)
if show_metrics:
    st.markdown("---")
    chatbot_dashboard()

# Footer
st.markdown("""
<div style='text-align:center; opacity:0.6; font-size:0.9em;'>
Built with <code>Streamlit</code> + your <code>code.py</code> hybrid logic.
</div>
""", unsafe_allow_html=True)
