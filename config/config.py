import os

# ==== LLM Provider Settings (Gemini default) ====
PROVIDER = os.getenv("PROVIDER", "gemini")  # "gemini" | "openai" | "groq" | "huggingface"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Gemini (default)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-lite")

# Hugging Face
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL   = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# ==== Search Settings ====
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "ddg")  # "ddg" or "serpapi"

# ==== RAG Settings ====
# Mode: "early" (pre-chunk entire corpus) or "late" (page-level index + late micro-chunking)
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "late")  # default to late

# Early chunking params
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Late chunking params
LATE_TOP_PAGES = int(os.getenv("LATE_TOP_PAGES", "4"))
LATE_MICRO_CHUNK_SIZE = int(os.getenv("LATE_MICRO_CHUNK_SIZE", "600"))
LATE_MICRO_OVERLAP = int(os.getenv("LATE_MICRO_OVERLAP", "150"))
LATE_FINAL_K = int(os.getenv("LATE_FINAL_K", "4"))
