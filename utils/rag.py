import io, re, os
from typing import List, Tuple
import faiss
import numpy as np
from config.config import (
    CHUNKING_MODE,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    LATE_TOP_PAGES, LATE_MICRO_CHUNK_SIZE, LATE_MICRO_OVERLAP, LATE_FINAL_K
)
from models.embeddings import Embeddings

try:
    import pypdf
except Exception:
    pypdf = None

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def pdf_pages_to_texts(file_bytes: bytes) -> List[str]:
    if pypdf is None:
        raise ImportError("pypdf is not installed")
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = _clean(text)
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L: break
        start = max(0, end - overlap)
    return chunks

def pseudo_pages_from_text(txt: str, page_chars: int = 4000) -> List[str]:
    txt = _clean(txt)
    return [txt[i:i+page_chars] for i in range(0, len(txt), page_chars)] if txt else []

class RAGIndex:
    def __init__(self):
        self.embedder = Embeddings()
        self.mode = CHUNKING_MODE.lower()
        self.early_chunks: List[str] = []
        self.early_index = None
        self.pages: List[str] = []
        self.page_index = None

    def build(self, docs: List[Tuple[str, bytes]]):
        if self.mode == "early":
            self._build_early(docs)
        else:
            self._build_late(docs)

    def _build_early(self, docs: List[Tuple[str, bytes]]):
        all_chunks = []
        for name, b in docs:
            try:
                if name.lower().endswith(".pdf"):
                    pages = pdf_pages_to_texts(b)
                    text = "\n\n".join(pages)
                else:
                    text = b.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            if not text.strip():
                continue
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            all_chunks.extend(chunks)
        if not all_chunks:
            raise ValueError("No text extracted from uploaded documents.")
        self.early_chunks = all_chunks
        embs = self.embedder.embed(all_chunks).astype("float32")
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        self.early_index = faiss.IndexFlatIP(d)
        self.early_index.add(embs)

    def _build_late(self, docs: List[Tuple[str, bytes]]):
        pages_all = []
        for name, b in docs:
            try:
                if name.lower().endswith(".pdf"):
                    pages = pdf_pages_to_texts(b)
                else:
                    txt = b.decode("utf-8", errors="ignore")
                    pages = pseudo_pages_from_text(txt, page_chars=4000)
            except Exception:
                pages = []
            pages_all.extend([_clean(p) for p in pages if p and _clean(p)])
        if not pages_all:
            raise ValueError("No text extracted from uploaded documents (late mode).")
        self.pages = pages_all
        embs = self.embedder.embed(self.pages).astype("float32")
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        self.page_index = faiss.IndexFlatIP(d)
        self.page_index.add(embs)

    def retrieve(self, query: str, top_k: int = TOP_K):
        if self.mode != "early":
            selected = self.late_retrieve(query, LATE_TOP_PAGES, LATE_FINAL_K)
            return [(1.0, t) for t in selected]
        if self.early_index is None:
            return []
        q = self.embedder.embed([query]).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.early_index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            results.append((float(score), self.early_chunks[idx]))
        return results

    def late_retrieve(self, query: str, top_pages: int = LATE_TOP_PAGES, final_k: int = LATE_FINAL_K):
        if self.page_index is None:
            return []
        q = self.embedder.embed([query]).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.page_index.search(q, top_pages)
        candidate_pages = [self.pages[idx] for idx in I[0] if idx != -1]

        micro_chunks = []
        for pg in candidate_pages:
            micro_chunks.extend(chunk_text(pg, LATE_MICRO_CHUNK_SIZE, LATE_MICRO_OVERLAP))
        if not micro_chunks:
            return []

        m_embs = self.embedder.embed(micro_chunks).astype("float32")
        faiss.normalize_L2(m_embs)
        d = m_embs.shape[1]
        local = faiss.IndexFlatIP(d)
        local.add(m_embs)
        D2, I2 = local.search(q, final_k)

        selected = [micro_chunks[i] for i in I2[0] if i != -1]
        return selected
