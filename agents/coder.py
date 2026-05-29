from __future__ import annotations
from .base import Agent
from config.prompts import CODER

class Coder(Agent):
    def __init__(self, engine):
        super().__init__(engine, CODER, "coder")

    async def code(self, plan, design, repo_ctx, full_files: str = ""):
        payload = (f"PLAN:\n{plan}\n\nDESIGN:\n{design}\n\n"
                   f"REPO_CONTEXT:\n{repo_ctx}\n\n")
        if full_files:
            payload += f"FULL_FILE_CONTENTS:\n{full_files}\n\n"
        payload += "Emit unified diffs, then JSON summary."
        return await self.run(payload, json_mode=False)
