from __future__ import annotations
from .base import Agent
from config.prompts import REPO_CTX

class RepoContextAgent(Agent):
    def __init__(self, engine):
        super().__init__(engine, REPO_CTX, "repo_ctx")

    async def analyze(self, retrieved):
        payload = "RETRIEVED:\n" + "\n---\n".join(
            f"{c['path']}\n{c['text']}" for c in retrieved)
        return await self.run(payload)
