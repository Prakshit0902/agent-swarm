from __future__ import annotations
import subprocess
from config.settings import settings
from .registry import tool

@tool("run_pytest","Run pytest in the workspace (or subpath).")
def run_pytest(path: str = ".", extra: str = "-q") -> dict:
    r = subprocess.run(["pytest", path, *extra.split()],
                       cwd=settings.workspace, capture_output=True, text=True,
                       timeout=settings.shell_timeout_s*4)
    # exit code 0 = all tests passed, 5 = no tests collected
    return {"ok": r.returncode in (0, 5), "rc": r.returncode,
            "stdout": r.stdout[-12000:], "stderr": r.stderr[-4000:]}
