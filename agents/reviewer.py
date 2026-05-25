from __future__ import annotations
from .base import Agent
from config.prompts import REVIEWER

class Reviewer(Agent):
    def __init__(self, engine):
        super().__init__(engine, REVIEWER, "reviewer")
