from __future__ import annotations

class Retriever:
    def __init__(self, store):
        self.store = store
    
    def retrieve(self, query: str, k: int = 8):
        return self.store.query(query, k=k)
