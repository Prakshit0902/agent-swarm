from __future__ import annotations
import asyncio, shlex, os
from config.settings import settings
from .registry import tool

ALLOWED_BIN = {"python","pytest","ruff","pyflakes","black","ls","cat","head","tail",
               "grep","rg","echo","wc","find","git","pip","make"}

@tool("shell", "Run an allow-listed shell command inside the sandbox.")
async def shell(cmd: str, cwd: str | None = None, timeout: int | None = None) -> dict:
    parts = shlex.split(cmd)
    if not parts or parts[0] not in ALLOWED_BIN:
        return {"ok": False, "stderr": f"binary not allowed: {parts[:1]}"}
    cwd = cwd or str(settings.workspace)
    if not cwd.startswith(tuple(settings.fs_allowed_roots)):
        return {"ok": False, "stderr": "cwd outside sandbox"}
    proc = await asyncio.create_subprocess_exec(
        *parts, cwd=cwd, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env={**os.environ, "PYTHONUNBUFFERED":"1"})
    try:
        out, err = await asyncio.wait_for(proc.communicate(),
                    timeout=timeout or settings.shell_timeout_s)
    except asyncio.TimeoutError:
        proc.kill(); return {"ok": False, "stderr": "timeout"}
    return {"ok": proc.returncode == 0, "rc": proc.returncode,
            "stdout": out.decode("utf-8","replace")[-8000:],
            "stderr": err.decode("utf-8","replace")[-4000:]}
