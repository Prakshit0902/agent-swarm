from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
import os

ROOT = Path(os.environ.get("KCS_ROOT", "/kaggle/working/kaggle_coder_swarm"))
ROOT.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    root: Path = ROOT
    workspace: Path = ROOT / "workspace"
    sandbox: Path  = ROOT / "sandbox"
    chroma_dir: Path = ROOT / "chroma"
    sessions_db: Path = ROOT / "sessions.db"
    cache_dir: Path = Path(os.environ.get("HF_HOME", "/kaggle/temp/hf"))

    # generation
    max_new_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05

    # swarm
    max_iterations: int = 6
    parallel_research: bool = True
    enable_web: bool = True

    # embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_tokens: int = 400
    chunk_overlap: int = 60

    # safety
    shell_timeout_s: int = 60
    fs_allowed_roots: tuple[str, ...] = (str(ROOT / "workspace"), str(ROOT / "sandbox"))

settings = Settings()
for p in (settings.workspace, settings.sandbox, settings.chroma_dir, settings.cache_dir):
    p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(settings.cache_dir))
os.environ.setdefault("HF_HUB_CACHE", str(settings.cache_dir))
os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.cache_dir))