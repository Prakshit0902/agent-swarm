from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Configure HF cache BEFORE any transformers / huggingface_hub import fires.
#
# Root cause: huggingface_hub reads HF_HOME at module-level import time.
# By the time backend_selector is used, any prior `import transformers` (e.g.
# via rag/store.py → sentence_transformers) has already locked the cache path
# to ~/.cache/huggingface — which on Kaggle is the ephemeral /root overlay.
#
# Fix: run _configure_hf_cache() at THIS module's import time (line below the
# function), before the `from .gpu_probe import ...` triggers any torch/HF code.
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def _configure_hf_cache() -> Path:
    """
    Point the HuggingFace cache at /kaggle/temp (or a suitable fallback).
    Called at module import time — before any downstream HF import.
    Returns the resolved HF home directory that will be used.
    """
    # Prefer /kaggle/temp — large, fast, not quota-gated like /kaggle/working
    candidates = [
        Path("/kaggle/temp"),
        Path("/tmp/hf_cache"),                          # fallback outside Kaggle
        Path.home() / ".cache" / "huggingface_kcs",    # last resort
    ]

    cache_root: Path | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.touch()
            probe.unlink()
            cache_root = candidate
            break
        except OSError:
            continue

    if cache_root is None:
        raise RuntimeError(
            "No writable HF cache directory found. "
            "Set HF_HOME to a writable path before importing this module."
        )

    hf_home = cache_root / "huggingface"
    hub_dir  = hf_home / "hub"
    tmp_dir  = cache_root / "tmp"

    for d in (hf_home, hub_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Set ALL the env vars that different HF library versions check
    env_updates = {
        "HF_HOME":               str(hf_home),
        "HF_HUB_CACHE":          str(hub_dir),
        "HUGGINGFACE_HUB_CACHE": str(hub_dir),
        "TRANSFORMERS_CACHE":    str(hub_dir),   # legacy compat
        # Redirect temp dir so shard extraction never lands in /kaggle/working
        "TMPDIR": str(tmp_dir),
        "TEMP":   str(tmp_dir),
        "TMP":    str(tmp_dir),
    }
    for k, v in env_updates.items():
        os.environ[k] = v

    # Defensive patch: if huggingface_hub was already imported, update its
    # module-level cache constants so new downloads still go to the right place.
    if "huggingface_hub" in sys.modules:
        try:
            import huggingface_hub.constants as _hf_const
            _hf_const.HF_HUB_CACHE = str(hub_dir)
            _hf_const.HUGGINGFACE_HUB_CACHE = str(hub_dir)
        except Exception:
            pass  # best-effort only

    # Warn if /kaggle/working is already polluted with stray model shards
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        try:
            used_gb = sum(
                f.stat().st_size
                for f in kaggle_working.rglob("*")
                if f.is_file()
            ) / (1024 ** 3)
            if used_gb > 5:
                _log.warning(
                    "/kaggle/working is %.1f GB — check for stray model shards. "
                    "Run: !find /kaggle/working -name '*.safetensors' -delete",
                    used_gb,
                )
        except Exception:
            pass

    _log.info("HF cache configured: %s", hf_home)
    return hf_home


# ── Run immediately at module import time — before any downstream HF import ───
_HF_CACHE_ROOT = _configure_hf_cache()
# ─────────────────────────────────────────────────────────────────────────────


# ── Now it is safe to import everything else ──────────────────────────────────
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
    "qwen25_coder_7b_bf16":   "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen25_coder_7b_awq":    "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "qwen25_coder_7b_gguf":   "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    "qwen25_coder_3b":        "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen25_coder_3b_gguf":   "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
    "qwen25_coder_1.5b":      "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen25_coder_1.5b_gguf": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
}


def choose(force: str | int | None = None) -> tuple[BackendChoice, GPUInfo]:
    g = probe()

    # ── Forced / CPU tier ─────────────────────────────────────────────────────
    if force in (3, "3", "cpu") or (force is None and not g.available):
        c = BackendChoice(
            "llamacpp", MODELS["qwen25_coder_1.5b_gguf"],
            "gguf-q4_k_m", 0, 8192, 0.0, "CPU/GGUF fallback",
        )
        banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
        return c, g

    if force in (1, "1"):
        c = BackendChoice(
            "vllm", MODELS["qwen25_coder_7b_bf16"], "bf16",
            tensor_parallel=max(1, g.count), max_model_len=65536,
            gpu_mem_util=0.90, notes="Forced Highest VRAM (vLLM BF16)",
        )
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

    # ── Auto-detect: Turing (T4) path ─────────────────────────────────────────
    # FIX: is_turing_only uses sum(vram) from gpu_probe, so dual T4 (29.12 GB)
    # correctly lands in the >= 28 GB branch, not the fallback.
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

    # ── Auto-detect: Ampere+ path — try vLLM ─────────────────────────────────
    if g.total_vram_gb >= 70 and g.bf16:
        c = BackendChoice(
            "vllm", MODELS["qwen25_coder_7b_bf16"], "bf16",
            tensor_parallel=g.count, max_model_len=65536,
            gpu_mem_util=0.90, notes="vLLM TP across Ampere+ GPUs",
        )
    elif g.total_vram_gb >= 24:
        c = BackendChoice(
            "vllm", MODELS["qwen25_coder_7b_awq"], "awq",
            tensor_parallel=g.count, max_model_len=32768,
            gpu_mem_util=0.88, notes="vLLM AWQ on Ampere+",
        )
    else:
        c = BackendChoice(
            "hf", MODELS["qwen25_coder_3b"], "bnb-nf4",
            tensor_parallel=1, max_model_len=16384,
            gpu_mem_util=0.90, notes="Low VRAM single GPU",
        )
    banner(g, c.backend, c.model_id, c.quantization, c.tensor_parallel)
    return c, g