from __future__ import annotations
from pathlib import Path
from config.settings import settings
from .registry import tool

def _safe(p: str) -> Path:
    rp = Path(p).resolve()
    if not str(rp).startswith(tuple(settings.fs_allowed_roots)):
        raise PermissionError(f"path outside sandbox: {rp}")
    return rp

@tool("read_file","Read a UTF-8 text file inside sandbox.")
def read_file(path: str, max_bytes: int = 200_000) -> str:
    p = _safe(path)
    return p.read_text("utf-8","replace")[:max_bytes]

@tool("write_file","Write/overwrite a text file inside sandbox.")
def write_file(path: str, content: str) -> dict:
    p = _safe(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8")
    return {"ok": True, "path": str(p), "bytes": len(content)}

@tool("list_dir","List a directory recursively (max 500 entries).")
def list_dir(path: str) -> list[str]:
    p = _safe(path); out = []
    for f in p.rglob("*"):
        out.append(str(f.relative_to(p)))
        if len(out) >= 500: break
    return out
