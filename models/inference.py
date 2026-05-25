from __future__ import annotations
from typing import Protocol, AsyncIterator, List, Dict
from .backend_selector import BackendChoice

ChatMessages = List[Dict[str, str]]

class LLMEngine(Protocol):
    async def chat(self, messages: ChatMessages, **kw) -> str: ...
    async def stream(self, messages: ChatMessages, **kw) -> AsyncIterator[str]: ...

def build_engine(choice: BackendChoice) -> LLMEngine:
    if choice.backend == "vllm":
        from .vllm_backend import VLLMEngine
        return VLLMEngine(choice)
    if choice.backend == "hf":
        from .hf_backend import HFEngine
        return HFEngine(choice)
    if choice.backend == "llamacpp":
        from .llamacpp_backend import LlamaCppEngine
        return LlamaCppEngine(choice)
    raise ValueError(choice.backend)
