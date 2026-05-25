from __future__ import annotations
import torch, subprocess, json
from dataclasses import dataclass, asdict

@dataclass
class GPUInfo:
    available: bool
    count: int
    names: list[str]
    vram_gb: list[float]
    cc_major: list[int]
    cc_minor: list[int]
    bf16: bool
    fp8: bool
    cuda_version: str | None
    is_turing_only: bool
    total_vram_gb: float

    def dict(self): return asdict(self)

def probe() -> GPUInfo:
    if not torch.cuda.is_available():
        return GPUInfo(False, 0, [], [], [], [], False, False, None, False, 0.0)
    n = torch.cuda.device_count()
    names, vram, ccmaj, ccmin = [], [], [], []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        names.append(p.name)
        vram.append(round(p.total_memory / 1024**3, 2))
        ccmaj.append(p.major); ccmin.append(p.minor)
    bf16 = all(m >= 8 for m in ccmaj)            # Ampere+
    fp8  = all((m, n) >= (8, 9) for m, n in zip(ccmaj, ccmin))  # Ada/Hopper
    is_turing_only = all(m == 7 for m in ccmaj)  # T4 / RTX20xx
    cuda_v = torch.version.cuda
    return GPUInfo(True, n, names, vram, ccmaj, ccmin, bf16, fp8, cuda_v,
                   is_turing_only, round(sum(vram), 2))

def banner(info: GPUInfo, backend: str, model: str, quant: str, tp: int):
    print("="*60)
    print(f"Detected GPUs   : {info.count} × {info.names}")
    print(f"VRAM            : {info.vram_gb} GB (total {info.total_vram_gb})")
    print(f"CUDA            : {info.cuda_version}")
    print(f"BF16 supported  : {info.bf16}")
    print(f"FP8 supported   : {info.fp8}")
    print(f"Chosen backend  : {backend}")
    print(f"Chosen model    : {model}")
    print(f"Quantization    : {quant}")
    print(f"Tensor parallel : {tp}")
    print("="*60)
