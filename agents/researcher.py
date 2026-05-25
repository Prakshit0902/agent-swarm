from __future__ import annotations
from .base import Agent
from config.prompts import RESEARCHER
from tools.web_search import web_search

class Researcher(Agent):
    def __init__(self, engine):
        super().__init__(engine, RESEARCHER, "researcher")

    async def investigate(self, query: str, k: int = 4):
        hits = await web_search(query, k=k)
        payload = "QUERY:\n"+query+"\n\nWEB_HITS:\n"+ "\n".join(
            f"- {h['title']} :: {h['url']}\n  {h['snippet']}" for h in hits)
        return await self.run(payload)
