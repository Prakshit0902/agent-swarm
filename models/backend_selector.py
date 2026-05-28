from __future__ import annotations
from dataclasses import dataclass
from .gpu_probe import probe, banner, GPUInfo

@dataclass
class BackendChoice:
    backend: str          # "vllm" | "hf" | "llamacpp"
    model_id: str
    quantization: str     # "bf16" | "fp16" | "awq" | "bnb-nf4" | "fp8" | "gguf-q4_k_m"
    tensor_parallel: int
    max_model_len: int
    gpu_mem_util: float
    notes: str

# Groq API cloud model mappings
MODELS = {
    "1": "llama-3.3-70b-versatile",
    "2": "llama-3.3-70b-versatile",
    "3": "llama-3.1-8b-instant",
}

def choose(force: str | int | None = None) -> tuple[BackendChoice, GPUInfo]:
    g = probe()
    import os

    tier = str(force) if force is not None else "1"
    if tier not in MODELS:
        tier = "1"

    model_id = os.environ.get("GROQ_MODEL", MODELS[tier])

    c = BackendChoice(
        backend="groq",
        model_id=model_id,
        quantization="none",
        tensor_parallel=1,
        max_model_len=8192,
        gpu_mem_util=0.0,
        notes=f"Groq API Cloud backend (Tier {tier} - model: {model_id})"
    )
    banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
    return c, g