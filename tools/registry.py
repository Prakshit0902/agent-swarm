from __future__ import annotations
from typing import Callable, Any
import inspect

REGISTRY: dict[str, dict] = {}

def tool(name: str, desc: str):
    def deco(fn: Callable):
        sig = inspect.signature(fn)
        REGISTRY[name] = {"fn": fn, "desc": desc, "params": str(sig)}
        return fn
    return deco
