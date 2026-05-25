from __future__ import annotations
import subprocess, tempfile, os
from config.settings import settings
from .registry import tool

@tool("apply_patch","Apply a unified diff to the workspace.")
def apply_patch(diff_text: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(diff_text); path = f.name
    try:
        r = subprocess.run(["git","apply","--whitespace=nowarn",path],
                           cwd=settings.workspace, capture_output=True, text=True)
        if r.returncode != 0:
            r2 = subprocess.run(["patch","-p1","-i",path],
                                cwd=settings.workspace, capture_output=True, text=True)
            return {"ok": r2.returncode == 0,
                    "stdout": r2.stdout, "stderr": r2.stderr}
        return {"ok": True, "stdout": r.stdout, "stderr": r.stderr}
    finally:
        os.unlink(path)
