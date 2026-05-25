from __future__ import annotations
import asyncio, torch, threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from config.settings import settings
from .backend_selector import BackendChoice

class HFEngine:
    def __init__(self, choice: BackendChoice):
        self.choice = choice
        kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if choice.quantization == "bnb-nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        # AWQ models carry their config; no extra kwargs needed.
        self.tokenizer = AutoTokenizer.from_pretrained(choice.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(choice.model_id, **kwargs)
        self.model.eval()
        self._lock = asyncio.Lock()

    def _render(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def chat(self, messages, max_new_tokens=None, temperature=None, **_):
        text = self._render(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or settings.max_new_tokens,
            do_sample=(temperature or settings.temperature) > 0,
            temperature=temperature or settings.temperature,
            top_p=settings.top_p, top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        async with self._lock:
            out = await asyncio.to_thread(self.model.generate, **inputs, **gen_kwargs)
        new = out[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    async def stream(self, messages, **kw):
        text = self._render(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=kw.get("max_new_tokens", settings.max_new_tokens),
            do_sample=settings.temperature > 0,
            temperature=settings.temperature,
            top_p=settings.top_p, top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        async with self._lock:
            thread.start()
            for tok in streamer:
                yield tok
            await asyncio.to_thread(thread.join)
