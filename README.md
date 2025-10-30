# ProcureCopilot — RAG + Live Web Search Chatbot for Infrastructure Tenders

**Use case**: Help Indian EPC/contracting teams (e.g., roads, bridges, metros) rapidly understand tender documents (NIT, BOQ, GCC/SCC, addenda), extract requirements, and answer queries. It also performs live web search on portals (e.g., eprocure, state PWD) to gather latest corrigenda or eligibility clarifications.

**Mandatory features implemented**
1. **RAG Integration** — PDF/TXT ingestion → chunking → embeddings (Sentence-Transformers) → FAISS index → top-k retrieval → context-aware LLM answer.
2. **Live Web Search** — Fallback/tooling via DuckDuckGo Search (no API key) or SerpAPI (optional) to fetch recent pages when RAG confidence is low or when user enables “Use Web”.
3. **Response Modes** — UI toggle **Concise** vs **Detailed**.

**Tech choices**
- **LLM**: OpenAI Chat Completions (configurable), easily swappable to Groq/Gemini.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local), so RAG works offline.
- **Vector DB**: FAISS in-memory (persist to `.faiss` optional).
- **UI**: Streamlit.
- **Search**: `duckduckgo-search` (no key) with optional SerpAPI.

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."           # required for LLM answers
# (optional)
export SERPAPI_KEY="..."                 # to use SerpAPI
streamlit run app.py
```

Upload one or more tender PDFs and ask questions like:
- *"What is the average annual turnover requirement?"*
- *"List EMD, bid security, and bid validity."*
- *"What are the plant & machinery requirements?"*
- *"Summarize eligibility: similar work definition, PQ criteria, and JV rules."*
- *"Any corrigendum after Oct 20, 2025?"* (toggle **Use Web**)
