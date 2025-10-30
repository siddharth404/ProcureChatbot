import io, os, re
from typing import List, Tuple
import faiss
import numpy as np
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from models.embeddings import Embeddings

try:
    import pypdf
except Exception:
    pypdf = None

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def pdf_to_text(file_bytes: bytes) -> str:
    if pypdf is None:
        raise ImportError("pypdf is not installed")
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = _clean(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

class RAGIndex:
    def __init__(self):
        self.embedder = Embeddings()
        self.index = None
        self.chunks: List[str] = []

    def build(self, docs: List[Tuple[str, bytes]]):
        # docs: List[(filename, bytes)]
        all_chunks = []
        for name, b in docs:
            try:
                if name.lower().endswith(".pdf"):
                    text = pdf_to_text(b)
                else:
                    text = b.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            if not text.strip():
                continue
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
        if not all_chunks:
            raise ValueError("No text extracted from uploaded documents.")
        self.chunks = all_chunks
        embs = self.embedder.embed(all_chunks).astype("float32")
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        # Normalize for cosine similarity
        faiss.normalize_L2(embs)
        self.index.add(embs)

    def retrieve(self, query: str, top_k: int = TOP_K):
        if self.index is None:
            return []
        q = self.embedder.embed([query]).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append((float(score), self.chunks[idx]))
        return results
