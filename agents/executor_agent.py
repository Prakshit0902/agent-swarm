from __future__ import annotations
from .base import Agent

class ExecutorAgent(Agent):
    def __init__(self, engine):
        super().__init__(engine, "ROLE: Executor; return a JSON of exec results.", "executor")
