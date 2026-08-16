"""Streamlit UI for the Meridian Supply Chain RAG system."""
from pathlib import Path
import streamlit as st

from ingest import ingest
from rag import answer_question, retrieve, get_collection

st.set_page_config(page_title="Meridian Supply Chain Assistant", page_icon="📦", layout="wide")

st.title("📦 Meridian Supply Chain Assistant")
st.caption("Retrieval-Augmented Generation over Meridian's supply-chain review and procurement policy.")

with st.sidebar:
    st.header("📄 Document Management")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected**")
        for f in uploaded_files:
            st.write(f"• {f.name}")

    if st.button("🔍 Index Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            paths = []
            for uploaded in uploaded_files:
                path = data_dir / Path(uploaded.name).name
                path.write_bytes(uploaded.getbuffer())
                paths.append(path)
            try:
                with st.spinner("Extracting, chunking, embedding, and storing..."):
                    result = ingest(paths)
                st.success(f"{result['files']} files processed, {result['chunks']} chunks stored.")
                get_collection.cache_clear()
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.subheader("📊 Collection")
    try:
        collection = get_collection()
        st.metric("Indexed chunks", collection.count())
    except Exception:
        st.info("No collection indexed yet. Upload the PDFs and click Index Documents.")

st.subheader("💬 Ask a Supply Chain Question")
question = st.text_area(
    "Question",
    placeholder="Example: Kaveri Metals recorded 88.1% on-time delivery and 1,150 PPM. Which policy clauses does this trigger?",
    height=110,
)

top_k = st.slider("Retrieved chunks", min_value=4, max_value=8, value=6, help="6 is a good default for cross-document questions.")

if st.button("🚀 Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = answer_question(question, top_k=top_k)
            st.subheader("💡 Answer")
            st.write(result["answer"])

            st.subheader("📚 Sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.markdown(f"- **{source['file']}** — Page **{source['page']}**")
            else:
                st.write("No sources found.")

            with st.expander("🔎 Retrieved chunks (debug)"):
                for i, chunk in enumerate(result.get("chunks", []), 1):
                    st.markdown(f"**{i}. {chunk['file']} — Page {chunk['page']} — distance {chunk['distance']:.4f}**")
                    st.write(chunk["text"])
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption("Meridian Components Pvt. Ltd. • Supply Chain RAG Assistant")
