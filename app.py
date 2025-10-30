import streamlit as st
from models.llm import LLMClient
from utils.rag import RAGIndex
from utils.search import web_search
from utils.prompting import build_messages
from config.config import CHUNKING_MODE, LATE_TOP_PAGES, LATE_FINAL_K

st.set_page_config(page_title="ProcureCopilot — Tenders RAG", page_icon="🦺", layout="wide")
st.title("🦺 ProcureCopilot — Tenders RAG + Live Search (Late Chunking Ready)")

with st.sidebar:
    st.header("Settings")
    st.caption(f"Chunking mode: **{CHUNKING_MODE.upper()}**")
    response_mode = st.radio("Response Mode", ["Concise", "Detailed"], index=0)
    use_web = st.toggle("Use Web Search", value=False)
    max_web = st.slider("Max Web Results", 0, 10, 5)
    st.caption("Upload PDFs/TXT and click 'Build Index'.")

uploaded = st.file_uploader("Upload tender documents", type=["pdf", "txt"], accept_multiple_files=True)

if "rag" not in st.session_state:
    st.session_state.rag = None

if st.button("🔧 Build Index"):
    if not uploaded:
        st.error("Please upload at least one document.")
    else:
        try:
            docs = [(f.name, f.read()) for f in uploaded]
            rag = RAGIndex()
            rag.build(docs)
            st.session_state.rag = rag
            count = len(rag.pages) if CHUNKING_MODE.lower() != "early" else len(rag.early_chunks)
            st.success(f"Index built. Units indexed: {count}.")
        except Exception as e:
            st.error(f"Index build failed: {e}")

query = st.text_input("Ask a question about the uploaded tenders")
if st.button("🚀 Ask"):
    if st.session_state.rag is None:
        st.warning("Build the index first.")
    elif not query.strip():
        st.warning("Type a question.")
    else:
        try:
            if CHUNKING_MODE.lower() == "early":
                retrieved = st.session_state.rag.retrieve(query)
                retrieved_for_prompt = retrieved
            else:
                selected = st.session_state.rag.late_retrieve(query, LATE_TOP_PAGES, LATE_FINAL_K)
                retrieved_for_prompt = [(1.0, t) for t in selected]

            web_results = []
            if use_web and max_web > 0:
                with st.spinner("Searching the web..."):
                    web_results = web_search(query, max_results=max_web)

            messages = build_messages(query, retrieved_for_prompt, response_mode, web_results)
            llm = LLMClient()

            with st.spinner("Generating answer..."):
                answer = llm.chat(messages)

            col1, col2 = st.columns([2,1])
            with col1:
                st.markdown("### Answer")
                st.write(answer)
            with col2:
                st.markdown("### Retrieved Context")
                if CHUNKING_MODE.lower() == "early":
                    for i, (score, chunk) in enumerate(retrieved_for_prompt, start=1):
                        with st.expander(f"Local #{i}"):
                            st.write(chunk[:1500] + ("..." if len(chunk) > 1500 else ""))
                else:
                    for i, (_, chunk) in enumerate(retrieved_for_prompt, start=1):
                        with st.expander(f"Context #{i}"):
                            st.write(chunk[:1500] + ("..." if len(chunk) > 1500 else ""))

                if web_results:
                    st.markdown("### Web Results")
                    for i, w in enumerate(web_results, start=1):
                        st.write(f"**Web #{i}:** [{w['title']}]({w['link']})")
                        if w.get('snippet'):
                            st.caption(w['snippet'])
        except Exception as e:
            st.error(f"Error: {e}")
