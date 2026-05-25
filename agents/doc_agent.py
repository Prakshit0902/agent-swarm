from __future__ import annotations
from .base import Agent

class DocAgent(Agent):
    def __init__(self, engine):
        super().__init__(engine, "ROLE: Documentation Agent.", "doc_agent")
