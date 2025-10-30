import streamlit as st
from typing import List, Tuple
from models.llm import LLMClient
from utils.rag import RAGIndex
from utils.search import web_search
from utils.prompting import build_messages

st.set_page_config(page_title="ProcureCopilot — Tenders RAG", page_icon="🦺", layout="wide")

st.title("🦺 ProcureCopilot — Tenders RAG + Live Search")

with st.sidebar:
    st.header("Settings")
    response_mode = st.radio("Response Mode", ["Concise", "Detailed"], index=0, help="Switch between brief vs in-depth answers.")
    use_web = st.toggle("Use Web Search", value=False, help="Include live web results.")
    max_web = st.slider("Max Web Results", 0, 10, 5)
    st.caption("Upload PDFs/TXT of tender docs below.")

uploaded = st.file_uploader("Upload tender documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if "rag" not in st.session_state:
    st.session_state.rag = None

build_clicked = st.button("🔧 Build Index")
if build_clicked:
    if not uploaded:
        st.error("Please upload at least one PDF/TXT document.")
    else:
        try:
            docs = [(f.name, f.read()) for f in uploaded]
            rag = RAGIndex()
            rag.build(docs)
            st.session_state.rag = rag
            st.success(f"Indexed {len(rag.chunks)} chunks.")
        except Exception as e:
            st.error(f"Index build failed: {e}")

query = st.text_input("Ask a question about the uploaded tenders")
go = st.button("🚀 Ask")

if go:
    if st.session_state.rag is None:
        st.warning("Build the index first.")
    elif not query.strip():
        st.warning("Type a question.")
    else:
        try:
            retrieved = st.session_state.rag.retrieve(query)
            web_results = []
            if use_web and max_web > 0:
                with st.spinner("Searching the web..."):
                    web_results = web_search(query, max_results=max_web)

            messages = build_messages(query, retrieved, response_mode, web_results)
            llm = LLMClient()
            with st.spinner("Generating answer..."):
                answer = llm.chat(messages)

            col1, col2 = st.columns([2,1])
            with col1:
                st.markdown("### Answer")
                st.write(answer)
            with col2:
                st.markdown("### Retrieved Chunks")
                for i, (score, chunk) in enumerate(retrieved, start=1):
                    with st.expander(f"Local #{i} (score={score:.3f})"):
                        st.write(chunk[:1500] + ("..." if len(chunk) > 1500 else ""))
                if web_results:
                    st.markdown("### Web Results")
                    for i, w in enumerate(web_results, start=1):
                        st.write(f"**Web #{i}:** [{w['title']}]({w['link']})")
                        if w.get('snippet'):
                            st.caption(w['snippet'])
        except Exception as e:
            st.error(f"Error: {e}")
