from __future__ import annotations
import subprocess
from config.settings import settings
from .registry import tool

@tool("ruff","Run ruff linter.")
def ruff(path: str = ".") -> dict:
    r = subprocess.run(["ruff","check",path], cwd=settings.workspace,
                       capture_output=True, text=True)
    return {"ok": r.returncode == 0, "stdout": r.stdout[-8000:]}
