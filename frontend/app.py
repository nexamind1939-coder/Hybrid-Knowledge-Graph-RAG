
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Hybrid Knowledge Graph RAG",
    page_icon="📚",
    layout="wide"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("📚 Hybrid Knowledge Graph RAG")
st.markdown("### Intelligent PDF Question Answering System")
st.caption("Streamlit | FastAPI | FAISS | Knowledge Graph | OpenAI")

# Backend status
st.subheader("Backend Status")

try:
    response = requests.get(
        f"{BACKEND_URL}/health",
        timeout=5
    )
    response.raise_for_status()
    st.success("Backend Connected")
except requests.RequestException:
    st.error("Backend connection failed. Check Docker.")
    st.stop()

st.divider()

# PDF upload
st.subheader("1. Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Selected {len(uploaded_files)} PDF file(s).")

    if st.button("Upload and Process PDFs", type="primary"):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf"
                            )
                        },
                        timeout=300
                    )

                    response.raise_for_status()
                    result = response.json()["result"]

                    st.success(f"Processed: {result['file_name']}")
                    st.write(f"Pages: {result['pages_processed']}")
                    st.write(f"Chunks: {result['chunks_created']}")

                except requests.RequestException as error:
                    st.error(
                        f"Failed to process {uploaded_file.name}: {error}"
                    )
                except (KeyError, ValueError) as error:
                    st.error(f"Unexpected response: {error}")

st.divider()

# Question section
st.subheader("2. Ask Your Documents")

question = st.text_input(
    "Enter your question",
    placeholder="What are the main concepts in my PDFs?"
)

if st.button("Ask Question"):
    if question.strip():
        st.info("Question answering will be implemented in the next stage.")
    else:
        st.warning("Please enter your question.")

st.divider()
st.caption("Answers will be based only on your uploaded PDF documents.")
