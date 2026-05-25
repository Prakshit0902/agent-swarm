from __future__ import annotations
from .base import Agent
from config.prompts import DEBUGGER

class Debugger(Agent):
    def __init__(self, engine):
        super().__init__(engine, DEBUGGER, "debugger")
