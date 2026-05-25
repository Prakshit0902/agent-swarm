from __future__ import annotations
from sentence_transformers import SentenceTransformer
from config.settings import settings

class Embedder:
    def __init__(self):
        self.m = SentenceTransformer(settings.embed_model, device="cpu")
    def embed(self, texts): return self.m.encode(texts, normalize_embeddings=True).tolist()
