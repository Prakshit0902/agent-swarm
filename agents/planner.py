from __future__ import annotations
from .base import Agent
from config.prompts import PLANNER

class Planner(Agent):
    def __init__(self, engine):
        super().__init__(engine, PLANNER, "planner")
