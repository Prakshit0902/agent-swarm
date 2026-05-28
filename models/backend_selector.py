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
    "qwen25_coder_7b_bf16": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen25_coder_7b_awq":  "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "qwen25_coder_7b_gguf": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "qwen25_coder_3b":      "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen25_coder_3b_gguf": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "qwen25_coder_1.5b":    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen25_coder_1.5b_gguf": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
}

def choose(force: str | int | None = None) -> tuple[BackendChoice, GPUInfo]:
    g = probe()

    if force in (3, "3", "cpu") or (force is None and not g.available):
        c = BackendChoice("llamacpp", MODELS["qwen25_coder_1.5b_gguf"],
                          "gguf-q4_k_m", 0, 8192, 0.0, "CPU/GGUF fallback")
        banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
        return c, g

    if force in (1, "1"):
        c = BackendChoice("vllm", MODELS["qwen25_coder_7b_bf16"], "bf16",
                          tensor_parallel=max(1, g.count), max_model_len=65536,
                          gpu_mem_util=0.90, notes="Forced Highest VRAM (vLLM BF16)")
        banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
        return c, g

    if force in (2, "2"):
        c = BackendChoice(
            "hf", MODELS["qwen25_coder_3b"], "bnb-nf4",
            tensor_parallel=1, max_model_len=16384, gpu_mem_util=0.90,
            notes="Forced Single GPU / Standard VRAM (HF 3B NF4)",
        )
        banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
        return c, g

    # T4 / Turing path
    if g.is_turing_only:
        if g.total_vram_gb >= 28:
            c = BackendChoice(
                "hf", MODELS["qwen25_coder_7b_awq"], "awq",
                tensor_parallel=g.count, max_model_len=32768,
                gpu_mem_util=0.92,
                notes="AWQ Qwen2.5-Coder-7B sharded across both T4s via device_map=auto",
            )
        else:
            c = BackendChoice(
                "hf", MODELS["qwen25_coder_3b"], "bnb-nf4",
                tensor_parallel=1, max_model_len=16384, gpu_mem_util=0.90,
                notes="Single T4 fallback: Qwen2.5-Coder-3B in NF4",
            )
        banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
        return c, g

    # Ampere+ path — try vLLM
    if g.total_vram_gb >= 70 and g.bf16:
        c = BackendChoice("vllm", MODELS["qwen25_coder_7b_bf16"], "bf16",
                          tensor_parallel=g.count, max_model_len=65536,
                          gpu_mem_util=0.90, notes="vLLM TP across Ampere+ GPUs")
    elif g.total_vram_gb >= 24:
        c = BackendChoice("vllm", MODELS["qwen25_coder_7b_awq"], "awq",
                          tensor_parallel=g.count, max_model_len=32768,
                          gpu_mem_util=0.88, notes="vLLM AWQ on Ampere+")
    else:
        c = BackendChoice("hf", MODELS["qwen25_coder_3b"], "bnb-nf4",
                          tensor_parallel=1, max_model_len=16384,
                          gpu_mem_util=0.90, notes="Low VRAM single GPU")
    banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
    return c, g