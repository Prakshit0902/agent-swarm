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
    s = RepoStore()
    n = s.index_repo(Path(repo))
    print(f"[green]Indexed {n} chunks[/green]")

@app.command()
def clone(url: str, dest: str = "repo"):
    print(git_clone(url, dest))

@app.command()
def solve(task: str, repo: str | None = None):
    """Run the swarm on a task."""
    if repo:
        s = RepoStore(); s.index_repo(Path(repo))
    sw = CodingSwarm()
    res = asyncio.run(sw.solve(task))
    print(res)

@app.command()
def info():
    from models.backend_selector import choose
    choose()

if __name__ == "__main__":
    app()
