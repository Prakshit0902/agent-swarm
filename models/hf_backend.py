from __future__ import annotations
import asyncio, os, torch, threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from config.settings import settings
from .backend_selector import BackendChoice


class HFEngine:
    def __init__(self, choice: BackendChoice):
        kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # ── Memory explosion guard ─────────────────────────────────────────────
        # Problem: AWQ from_pretrained() creates a full-precision CPU staging
        # buffer for every shard before quantising. On dual T4 with the 7B AWQ
        # model (4 shards × ~12 GB each), peak CPU RAM reaches ~48 GB — far
        # above Kaggle's 14 GB CPU limit, causing an OOM kernel crash.
        #
        # Fix:
        #   max_memory  — caps each GPU to 90% of its VRAM so HF won't over-
        #                 commit. Layers that genuinely don't fit spill to the
        #                 offload path instead of being staged in CPU RAM.
        #   offload_folder — spill-to-disk instead of spill-to-RAM. Peak CPU
        #                 RAM drops from ~48 GB to ~2–4 GB for the 7B AWQ model.
        # ──────────────────────────────────────────────────────────────────────
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count > 0:
            per_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            # Use the requested gpu_mem_util ratio to cap each GPU
            cap_gb = max(1, int(per_gpu_gb * choice.gpu_mem_util))
            kwargs["max_memory"] = {i: f"{cap_gb}GiB" for i in range(gpu_count)}

            # offload_folder: disk-based fallback for layers that don't fit VRAM
            # Points to /kaggle/temp/offload (set by backend_selector._configure_hf_cache)
            _offload = os.path.join(
                os.environ.get("TMPDIR", "/kaggle/temp"), "offload"
            )
            os.makedirs(_offload, exist_ok=True)
            kwargs["offload_folder"] = _offload

        # BitsAndBytes 4-bit quantisation config
        if choice.quantization == "bnb-nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        # AWQ models carry their own quantisation config inside model files;
        # no extra BitsAndBytesConfig needed — device_map + max_memory suffices.

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
