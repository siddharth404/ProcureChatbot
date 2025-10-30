# 🦺 ProcureCopilot — RAG + Live Web Search Chatbot for Infrastructure Tenders

<p align="center">
  <a href="https://streamlit.io/cloud"><img alt="Deploy on Streamlit" src="https://img.shields.io/badge/Deploy-Streamlit%20Cloud-brightgreen"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-informational">
  <img alt="RAG" src="https://img.shields.io/badge/RAG-FAISS%20%2B%20SentenceTransformer-green">
  <img alt="LLMs" src="https://img.shields.io/badge/LLMs-OpenAI%20%7C%20Groq%20%7C%20Gemini-orange">
</p>

---

### 📸 Screenshot
> Replace `docs/screenshot.png` with your app screenshot after deployment.

<p align="center">
  <img src="docs/screenshot.png" alt="ProcureCopilot UI" width="800"/>
</p>

---

## 🧠 Project Overview

**ProcureCopilot** is an AI-powered assistant designed for **infrastructure tender analysis** (EPC / contracting domain).  
It helps engineers, managers, and analysts **understand tender documents**, extract eligibility criteria, and monitor corrigenda — all through natural language queries.

### 💼 Use Case
Indian EPC firms often deal with 100+ page tender documents (NIT, GCC/SCC, BOQ, addenda).  
ProcureCopilot lets them:
- Upload PDF or TXT tender documents
- Ask queries like “What is the turnover requirement?” or “List EMD and bid validity period.”
- Perform **live web searches** for corrigenda or clarifications.
- Choose between **Concise** and **Detailed** response styles.

---

## 🧩 Key Features

| Feature | Description |
|----------|--------------|
| **RAG Integration** | Ingest and index PDFs using SentenceTransformer embeddings + FAISS for retrieval. |
| **Live Web Search** | Queries live data (DuckDuckGo or SerpAPI). Useful for corrigenda, updates, or bidder clarifications. |
| **Concise/Detailed Modes** | UI toggle to control verbosity of responses. |
| **Multi-LLM Support** | Works with OpenAI (default), Groq, or Gemini with a single environment variable change. |
| **Error Handling** | Graceful fallbacks, clear user messages. |
| **Streamlit UI** | Intuitive and responsive interface for document upload and Q&A. |

---

## ⚙️ Architecture Overview

```bash
ProcureCopilot/
├── app.py                     # Streamlit UI, user interactions
├── config/
│   └── config.py              # Centralized configuration and env vars
├── models/
│   ├── llm.py                 # Unified LLM adapter (OpenAI, Groq, Gemini)
│   └── embeddings.py          # SentenceTransformer-based embedding class
├── utils/
│   ├── rag.py                 # RAG engine: PDF extraction, chunking, FAISS
│   ├── search.py              # DuckDuckGo & SerpAPI-based live search
│   └── prompting.py           # Prompt construction for Concise/Detailed modes
├── docs/
│   └── screenshot.png         # Placeholder for app UI screenshot
├── requirements.txt
└── README.md
```
🚀 Quickstart
1️⃣ Clone and Install Dependencies
git clone https://github.com/<your-username>/ProcureCopilot.git
cd ProcureCopilot
pip install -r requirements.txt

2️⃣ Configure Environment Variables

Create a .env file or export variables directly:

# Provider: openai (default) | groq | gemini
export PROVIDER=openai

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"

# Groq (optional)
export GROQ_API_KEY="gsk_..."
export GROQ_MODEL="llama-3.1-70b-versatile"

# Gemini (optional)
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-1.5-flash"

# Search
export SEARCH_PROVIDER="ddg"   # or "serpapi"
export SERPAPI_KEY=""           # optional, if using SerpAPI

# RAG Settings
export EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
export CHUNK_SIZE=1200
export CHUNK_OVERLAP=200
export TOP_K=4

🧰 Running Locally
streamlit run app.py


Visit 👉 http://localhost:8501 in your browser.

Steps:

Upload one or more tender PDFs or TXT files.

Click 🔧 Build Index (it will chunk & embed text).

Type a query like:

“Summarize eligibility criteria.”

“What is the EMD and bid validity period?”

(Optional) Toggle Use Web Search to pull live results.

Switch Response Mode to “Concise” or “Detailed” as needed.

☁️ Deploying on Streamlit Cloud

Push your repo to GitHub.

Go to Streamlit Cloud
.

Create a New App → Select your GitHub repo.

Main file: app.py

Branch: main

In Secrets, add:

OPENAI_API_KEY = "sk-..."
SERPAPI_KEY = ""
GROQ_API_KEY = ""
GEMINI_API_KEY = ""


Click Deploy 🚀

Copy your public URL (e.g. https://yourname-procurecopilot.streamlit.app)
→ Add it in your presentation or submission.

🧮 How It Works
Step	Description
1. Ingestion	PDFs are parsed using pypdf → text cleaned & chunked.
2. Embedding	Chunks embedded via SentenceTransformer.
3. Indexing	FAISS vector store for similarity search.
4. Retrieval	Top-K chunks retrieved based on query.
5. Augmentation	LLM combines retrieved + web context → final answer.
6. Output Modes	Prompt builder formats response as Concise or Detailed.
🧪 Example Queries
Query	Expected Behavior
“List the eligibility and turnover criteria.”	Extracts relevant clauses from local PDF.
“What is the EMD amount and bid validity?”	Returns numeric details from the tender.
“Has any corrigendum been published recently?”	Triggers web search for latest results.
“Summarize all technical requirements.”	Detailed structured response with bullet points.
🔧 Configuration Parameters
Variable	Default	Description
CHUNK_SIZE	1200	Token length per text chunk
CHUNK_OVERLAP	200	Overlap between chunks
TOP_K	4	Retrieved chunks count
SEARCH_PROVIDER	ddg	ddg or serpapi
PROVIDER	openai	LLM provider: openai, groq, gemini
🧩 Technology Stack
Layer	Library
Frontend / UI	Streamlit
LLM API	OpenAI / Groq / Gemini
Embeddings	Sentence-Transformers
Vector Search	FAISS
PDF Parsing	PyPDF
Web Search	DuckDuckGo / SerpAPI
🧱 Project Strengths

✅ Modular and extensible design
✅ Streamlit-ready for instant deployment
✅ Secure key handling (env vars only)
✅ RAG + Web Search dual mode
✅ Domain-specific tender analysis focus
✅ Multi-provider support (OpenAI, Groq, Gemini)

💻 Requirements
streamlit==1.37.0
openai>=1.30.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
pypdf>=4.3.1
duckduckgo-search>=6.2.11
serpapi>=0.1.5
groq>=0.9.0
google-generativeai>=0.7.0
Pillow>=10.4.0

🌍 Example Deployment (Preview)

Live App:

(Replace with your deployed Streamlit Cloud URL)
https://<your-username>-procurecopilot.streamlit.app

🧾 License

This project is licensed under the MIT License
.

MIT License © 2025 Siddharth Kaushik  
Developed under NeoStats AI Engineer Case Study

🤝 Contributing

Contributions are welcome!
Please fork this repository and submit a pull request for improvements or bug fixes.

👤 Author

Siddharth Kaushik
M.Sc. Data Science & Management — IIT Ropar × IIM Amritsar
📧 Email me

<p align="center">Built with ❤️ using Streamlit, FAISS, and modern LLMs.</p> ```
