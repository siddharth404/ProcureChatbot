from typing import List
from config.config import EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
