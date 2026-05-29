#!/usr/bin/env python
# ============================================================
# CELL 1 — Paste this as the VERY FIRST cell in your Kaggle
#           notebook. Nothing else must run before this.
# ============================================================

"""
Kaggle T4 x2 bootstrap: sets up HF cache routing and validates
the GPU tier before any model or framework import happens.

Why this must be first:
  huggingface_hub reads HF_HOME at import time (module-level code).
  If transformers is imported before you set the env var, the default
  ~/.cache path is baked in for the entire session.
"""

import os, sys, tempfile
from pathlib import Path

# ── 1. Cache routing ──────────────────────────────────────────
# /kaggle/temp  → large disk, not quota-gated (not /kaggle/working)
HF_CACHE = Path("/kaggle/temp/huggingface")
HF_HUB   = HF_CACHE / "hub"
TMPDIR   = Path("/kaggle/temp/tmp")
OFFLOAD  = Path("/kaggle/temp/offload")

for d in [HF_CACHE, HF_HUB, TMPDIR, OFFLOAD]:
    d.mkdir(parents=True, exist_ok=True)

# Every env var the HF ecosystem checks (across all versions)
for var, val in {
    "HF_HOME":               str(HF_CACHE),
    "HF_HUB_CACHE":          str(HF_HUB),
    "HUGGINGFACE_HUB_CACHE": str(HF_HUB),
    "TRANSFORMERS_CACHE":    str(HF_HUB),
    "TMPDIR": str(TMPDIR), "TEMP": str(TMPDIR), "TMP": str(TMPDIR),
    "HF_HUB_DISABLE_XET": "1",          # disable XET protocol (saves RAM)
    "TOKENIZERS_PARALLELISM": "false",   # avoid tokenizer fork warnings
}.items():
    os.environ[var] = val

# Also patch Python's own tempfile module
tempfile.tempdir = str(TMPDIR)

print("✅ HF cache  →", str(HF_CACHE))
print("✅ HF hub    →", str(HF_HUB))
print("✅ TMPDIR    →", str(TMPDIR))
print("✅ offload   →", str(OFFLOAD))

# ── 2. Verify GPU layout ──────────────────────────────────────
import torch

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    per_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
    total   = per_gpu * n
    arch    = {9: "Hopper", 8: "Ampere", 7: "Turing"}.get(
                  torch.cuda.get_device_properties(0).major, "Unknown")
    print(f"\n✅ {n}× {torch.cuda.get_device_properties(0).name}")
    print(f"   Per-GPU VRAM : {per_gpu:.1f} GB")
    print(f"   Total VRAM   : {total:.1f} GB   ← backend_selector uses THIS")
    print(f"   Architecture : {arch}")

    # Tier prediction
    if arch == "Turing" and total >= 28:
        print("   → Tier 2 (HF Transformers + AWQ) ✓")
    elif total > 70:
        print("   → Tier 1 (vLLM BF16) ✓")
    else:
        print(f"   → Tier 3 (HF 4-bit bnb) — only {total:.1f} GB total")
else:
    print("\n⚠️  No CUDA — will use CPU tier (Llama.cpp)")

# ── 3. Disk space check ───────────────────────────────────────
import shutil

print()
for label, path in [("working", "/kaggle/working"), ("temp", "/kaggle/temp")]:
    try:
        total_d, used_d, free_d = shutil.disk_usage(path)
        print(f"   /kaggle/{label}: {used_d/1e9:.1f} GB used / {total_d/1e9:.1f} GB total")
        if label == "working" and used_d / 1e9 > 5:
            print("   ⚠️  /kaggle/working is large — stray model shards detected!")
            print("       Run: !find /kaggle/working -name '*.safetensors' -delete")
    except FileNotFoundError:
        pass
