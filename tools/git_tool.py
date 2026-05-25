from __future__ import annotations
import git
from config.settings import settings
from .registry import tool

@tool("git_clone","Clone a repo into the workspace.")
def git_clone(url: str, dest: str = "repo") -> dict:
    target = settings.workspace / dest
    if target.exists(): return {"ok": True, "path": str(target), "note":"exists"}
    git.Repo.clone_from(url, target)
    return {"ok": True, "path": str(target)}

@tool("git_status","Get git status of workspace.")
def git_status(repo_path: str = "repo") -> str:
    r = git.Repo(settings.workspace / repo_path)
    return r.git.status()
