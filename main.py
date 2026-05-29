from __future__ import annotations
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Disable chromadb telemetry which fails on Kaggle
import asyncio, typer, warnings
from pathlib import Path
from rich import print

# Suppress verbose terminal spam from PyTorch, Transformers, bitsandbytes, and DuckDuckGo
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from config.settings import settings
from execution.swarm import CodingSwarm
from rag.store import RepoStore
from tools.git_tool import git_clone

app = typer.Typer()

def resolve_path(p_str: str) -> Path:
    p = Path(p_str)
    if p.exists() or p.is_absolute():
        return p
    # Try resolving relative to settings.workspace (e.g. settings.workspace / "workspace/repo")
    alt = settings.workspace / p_str
    if alt.exists():
        return alt
    # Fallback: if user passed 'workspace/repo', check settings.workspace / 'repo'
    if p_str.startswith("workspace/"):
        alt_strip = settings.workspace / p_str.replace("workspace/", "", 1)
        if alt_strip.exists():
            return alt_strip
    return p

@app.command()
def index(repo: str = "."):
    """Index a local repo for RAG."""
    repo_path = resolve_path(repo)
    s = RepoStore()
    n = s.index_repo(repo_path)
    print(f"[green]Indexed {n} chunks from {repo_path}[/green]")

@app.command()
def clone(url: str, dest: str = "repo"):
    print(git_clone(url, dest))

@app.command()
def solve(
    task: str,
    repo: str | None = None,
    tier: int | None = typer.Option(
        None,
        "--tier",
        "-t",
        help="Forced model tier to use: 1 = Highest VRAM (vLLM BF16), 2 = Single GPU (HF 3B NF4), 3 = CPU fallback (Llama.cpp 1.5B GGUF)"
    )
):
    """Run the swarm on a task."""
    if repo:
        repo_path = resolve_path(repo)
        s = RepoStore(); s.index_repo(repo_path)
    sw = CodingSwarm(model_tier=tier)
    res = asyncio.run(sw.solve(task))
    print(res)

@app.command()
def info(
    tier: int | None = typer.Option(
        None,
        "--tier",
        "-t",
        help="Forced model tier to inspect: 1, 2, or 3"
    )
):
    from models.backend_selector import choose
    choose(force=tier)

if __name__ == "__main__":
    app()
