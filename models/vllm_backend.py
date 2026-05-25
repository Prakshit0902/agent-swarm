from __future__ import annotations
import asyncio
from .backend_selector import BackendChoice
from config.settings import settings

class VLLMEngine:
    def __init__(self, choice: BackendChoice):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # noqa
        args = AsyncEngineArgs(
            model=choice.model_id,
            tensor_parallel_size=choice.tensor_parallel,
            dtype="bfloat16" if choice.quantization == "bf16" else "auto",
            quantization=None if choice.quantization in ("bf16", "fp8") else choice.quantization,
            gpu_memory_utilization=choice.gpu_mem_util,
            max_model_len=choice.max_model_len,
            enforce_eager=False,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(choice.model_id, trust_remote_code=True)
        self.SamplingParams = SamplingParams

    def _render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    async def chat(self, messages, **kw) -> str:
        sp = self.SamplingParams(
            temperature=kw.get("temperature", settings.temperature),
            top_p=settings.top_p, top_k=settings.top_k,
            max_tokens=kw.get("max_new_tokens", settings.max_new_tokens),
            repetition_penalty=settings.repetition_penalty,
        )
        prompt = self._render(messages)
        req_id = f"req-{id(messages)}-{asyncio.get_event_loop().time()}"
        final = ""
        async for out in self.engine.generate(prompt, sp, request_id=req_id):
            final = out.outputs[0].text
        return final

    async def stream(self, messages, **kw):
        sp = self.SamplingParams(
            temperature=kw.get("temperature", settings.temperature),
            top_p=settings.top_p, max_tokens=kw.get("max_new_tokens", settings.max_new_tokens),
        )
        prompt = self._render(messages)
        req_id = f"req-{id(messages)}-{asyncio.get_event_loop().time()}"
        prev = ""
        async for out in self.engine.generate(prompt, sp, request_id=req_id):
            cur = out.outputs[0].text
            delta = cur[len(prev):]; prev = cur
            if delta: yield delta
