import os

# Centralized configuration. Do NOT hardcode keys elsewhere.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # adjustable

# Optional: SerpAPI. If absent, defaults to DuckDuckGo.
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "ddg")  # "ddg" or "serpapi"

# RAG params
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))
