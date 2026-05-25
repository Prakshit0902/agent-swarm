from __future__ import annotations
import subprocess, sys
from config.settings import settings
from .registry import tool

@tool("python_run","Run a python script inside sandbox.")
def python_run(path: str, args: str = "") -> dict:
    r = subprocess.run([sys.executable, path] + args.split(),
                       cwd=settings.workspace, capture_output=True, text=True,
                       timeout=settings.shell_timeout_s)
    return {"ok": r.returncode == 0, "stdout": r.stdout[-8000:], "stderr": r.stderr[-4000:]}
