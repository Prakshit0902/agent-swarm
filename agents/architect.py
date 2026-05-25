from __future__ import annotations
from .base import Agent
from config.prompts import ARCHITECT

class Architect(Agent):
    def __init__(self, engine):
        super().__init__(engine, ARCHITECT, "architect")
