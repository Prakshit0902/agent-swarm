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

# Verified IDs (checked on HF Hub at build time)
MODELS = {
    "qwen3_coder_30b_bf16": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen3_coder_30b_fp8":  "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "qwen3_coder_30b_awq":  "AIDXteam/Qwen3-Coder-30B-A3B-Instruct-AWQ",
    "qwen25_coder_7b":      "Qwen/Qwen2.5-Coder-7B-Instruct",
    "gemma3_27b":           "google/gemma-3-27b-it",   # correction: gemma-4-27b-it does NOT exist
    "gemma2_9b":            "google/gemma-2-9b-it",    # correction: gemma-4-9b-it does NOT exist
    "qwen25_coder_7b_gguf": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
}

def choose(force: str | None = None) -> tuple[BackendChoice, GPUInfo]:
    g = probe()
    notes = []

    if force == "cpu" or not g.available:
        c = BackendChoice("llamacpp", MODELS["qwen25_coder_7b_gguf"],
                          "gguf-q4_k_m", 0, 8192, 0.0, "CPU/GGUF fallback")
        banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
        return c, g

    # T4 / Turing path — vLLM is unreliable for qwen3_moe on sm_75
    if g.is_turing_only:
        if g.total_vram_gb >= 28:
            c = BackendChoice(
                "hf", MODELS["qwen3_coder_30b_awq"], "awq",
                tensor_parallel=g.count, max_model_len=32768,
                gpu_mem_util=0.92,
                notes="AWQ Qwen3-Coder-30B-A3B sharded across both T4s via device_map=auto",
            )
        else:
            c = BackendChoice(
                "hf", MODELS["qwen25_coder_7b"], "bnb-nf4",
                tensor_parallel=1, max_model_len=16384, gpu_mem_util=0.90,
                notes="Single T4 fallback: Qwen2.5-Coder-7B in NF4",
            )
        banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
        return c, g

    # Ampere+ path — try vLLM
    if g.total_vram_gb >= 70 and g.bf16:
        model = MODELS["qwen3_coder_30b_fp8"] if g.fp8 else MODELS["qwen3_coder_30b_bf16"]
        c = BackendChoice("vllm", model, "fp8" if g.fp8 else "bf16",
                          tensor_parallel=g.count, max_model_len=65536,
                          gpu_mem_util=0.90, notes="vLLM TP across Ampere+ GPUs")
    elif g.total_vram_gb >= 24:
        c = BackendChoice("vllm", MODELS["qwen3_coder_30b_awq"], "awq",
                          tensor_parallel=g.count, max_model_len=32768,
                          gpu_mem_util=0.88, notes="vLLM AWQ on Ampere+")
    else:
        c = BackendChoice("hf", MODELS["qwen25_coder_7b"], "bnb-nf4",
                          tensor_parallel=1, max_model_len=16384,
                          gpu_mem_util=0.90, notes="Low VRAM single GPU")
    banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
    return c, g
