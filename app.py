import streamlit as st
import tempfile
import os
from agent import create_agent
from rag_engine import ingest_document

st.set_page_config(page_title="SmartResearch", page_icon="🧠", layout="wide")
st.title("SmartResearch — AI Research Assistant")
st.markdown("Powered by LLaMA 3.2 (local) + ChromaDB + LangChain")

# Initialize agent once per browser session
if "agent" not in st.session_state:
    with st.spinner("Loading AI model (first load takes 30 seconds)..."):
        st.session_state.agent = create_agent()
        st.session_state.messages = []

# Sidebar: document upload
with st.sidebar:
    st.header("Document Management")
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded and st.button("Process Document"):
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getvalue())
            tmp_path = f.name

        with st.spinner(f"Processing {uploaded.name}..."):
            n = ingest_document(tmp_path)
            os.unlink(tmp_path)
            st.success(f"Indexed {n} chunks from {uploaded.name}")

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent = create_agent()
        st.rerun()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask anything about your documents or any topic..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking (30-60 seconds on CPU)..."):
            try:
                result = st.session_state.agent.invoke({"input": question})
                answer = result.get("output", "I could not generate a response.")
            except Exception as e:
                answer = f"An error occurred: {str(e)}. Try rephrasing your question."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
