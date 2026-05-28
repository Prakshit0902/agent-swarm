from __future__ import annotations
from typing import Protocol, AsyncIterator, List, Dict
from .backend_selector import BackendChoice

ChatMessages = List[Dict[str, str]]

class LLMEngine(Protocol):
    async def chat(self, messages: ChatMessages, **kw) -> str: ...
    async def stream(self, messages: ChatMessages, **kw) -> AsyncIterator[str]: ...

def build_engine(choice: BackendChoice) -> LLMEngine:
    if choice.backend == "groq":
        from .groq_backend import GroqEngine
        return GroqEngine(choice)
    if choice.backend == "vllm":
        from .vllm_backend import VLLMEngine
        return VLLMEngine(choice)
    if choice.backend == "hf":
        from .hf_backend import HFEngine
        return HFEngine(choice)
    if choice.backend == "llamacpp":
        try:
            from .llamacpp_backend import LlamaCppEngine
            return LlamaCppEngine(choice)
        except (ImportError, RuntimeError) as e:
            from rich import print
            print("[yellow]llama-cpp-python is not installed. Falling back to Hugging Face CPU/GPU inference with Qwen2.5-Coder-1.5B-Instruct.[/yellow]")
            from .backend_selector import BackendChoice
            fallback_choice = BackendChoice(
                backend="hf",
                model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                quantization="none",
                tensor_parallel=1,
                max_model_len=8192,
                gpu_mem_util=0.90,
                notes="Hugging Face 1.5B Fallback (llama-cpp-python missing)"
            )
            from .hf_backend import HFEngine
            return HFEngine(fallback_choice)
    raise ValueError(choice.backend)
