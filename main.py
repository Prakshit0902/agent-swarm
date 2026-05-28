from __future__ import annotations
import asyncio, typer
from pathlib import Path
from rich import print
from config.settings import settings
from execution.swarm import CodingSwarm
from rag.store import RepoStore
from tools.git_tool import git_clone

app = typer.Typer()

@app.command()
def index(repo: str = "."):
    """Index a local repo for RAG."""
    from config.settings import resolve_repo_path
    s = RepoStore()
    resolved = resolve_repo_path(repo)
    n = s.index_repo(resolved)
    print(f"[green]Indexed {n} chunks[/green]")

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
    resolved_repo = None
    if repo:
        from config.settings import resolve_repo_path
        resolved_repo = resolve_repo_path(repo)
        s = RepoStore(); s.index_repo(resolved_repo)
    sw = CodingSwarm(repo_root=resolved_repo, model_tier=tier)
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
