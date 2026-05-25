from __future__ import annotations
import asyncio
from .backend_selector import BackendChoice

class LlamaCppEngine:
    def __init__(self, choice: BackendChoice):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("pip install llama-cpp-python required for CPU fallback") from e
        # auto-download a Q4_K_M GGUF
        self.llm = Llama.from_pretrained(
            repo_id=choice.model_id, filename="*Q4_K_M.gguf",
            n_ctx=choice.max_model_len, n_threads=4, verbose=False,
        )

    async def chat(self, messages, **kw):
        def _run():
            r = self.llm.create_chat_completion(messages=messages,
                                                 temperature=kw.get("temperature", 0.2),
                                                 max_tokens=kw.get("max_new_tokens", 1024))
            return r["choices"][0]["message"]["content"]
        return await asyncio.to_thread(_run)

    async def stream(self, messages, **kw):
        for chunk in self.llm.create_chat_completion(messages=messages, stream=True,
                                                     max_tokens=kw.get("max_new_tokens", 1024)):
            d = chunk["choices"][0]["delta"].get("content", "")
            if d: yield d
