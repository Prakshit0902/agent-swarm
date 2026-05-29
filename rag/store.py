from __future__ import annotations
import chromadb, hashlib
# Monkey patch chromadb's telemetry client to completely nullify the broken capture() method
try:
    import chromadb.telemetry.posthog
    class DummyClient:
        def capture(self, *args, **kwargs): pass
    chromadb.telemetry.posthog.Posthog = DummyClient
except Exception:
    pass

from config.settings import settings
from .embedder import Embedder
from .chunker import chunk_file
from pathlib import Path

class RepoStore:
    def __init__(self, collection: str = "repo"):
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.col = self.client.get_or_create_collection(collection)
        self.embed = Embedder()

    def index_repo(self, root: Path, exts=(".py",".md",".js",".ts",".go",".rs",".java",".cpp",".c",".yaml",".toml",".json",".csv",".txt",".html",".xml",".sql",".sh")):
        docs, ids, metas = [], [], []
        for p in root.rglob("*"):
            if ".git" in p.parts:
                continue
            if not p.is_file() or p.suffix.lower() not in exts or p.stat().st_size > 500_000:
                continue
            for ch in chunk_file(p):
                h = hashlib.sha1((ch["path"]+str(ch["start"])+ch["text"][:64]).encode()).hexdigest()
                docs.append(ch["text"]); ids.append(h)
                metas.append({"path": ch["path"], "lang": ch["lang"], "start": ch["start"]})
        if not docs: return 0
        embs = self.embed.embed(docs)
        # upsert in batches
        B = 256
        for i in range(0, len(docs), B):
            self.col.upsert(ids=ids[i:i+B], documents=docs[i:i+B],
                            embeddings=embs[i:i+B], metadatas=metas[i:i+B])
        return len(docs)

    def query(self, q: str, k: int = 8) -> list[dict]:
        e = self.embed.embed([q])[0]
        r = self.col.query(query_embeddings=[e], n_results=k)
        return [{"path": m["path"], "text": d, "score": s}
                for d, m, s in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
