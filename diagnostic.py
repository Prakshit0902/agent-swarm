#!/usr/bin/env python
"""
diagnostic.py — Run this in a Kaggle cell to diagnose the current state
of the HF cache routing WITHOUT resetting anything.

Paste the contents into a cell and run it to see:
  - Where HF is currently writing models
  - What's eating /kaggle/working disk
  - Whether your GPU tier is being detected correctly
"""

import os, shutil, subprocess
from pathlib import Path

print("=" * 60)
print("HF ENVIRONMENT VARIABLES")
print("=" * 60)
for var in [
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE", "TMPDIR", "TEMP", "TMP"
]:
    val = os.environ.get(var, "<not set>")
    bad = "/kaggle/working" in val
    flag = "  ⚠️  POINTS TO /kaggle/working — THIS IS THE BUG" if bad else ""
    print(f"  {var:28s} = {val}{flag}")

print()
print("=" * 60)
print("DISK USAGE")
print("=" * 60)
for path in ["/kaggle/working", "/kaggle/temp", "/root/.cache", "/tmp"]:
    p = Path(path)
    if not p.exists():
        print(f"  {path}: <does not exist>")
        continue
    try:
        total, used, free = shutil.disk_usage(path)
        print(f"  {path}: {used/1e9:.2f} GB used / {total/1e9:.1f} GB total")
    except Exception as e:
        print(f"  {path}: error — {e}")

# Show the 10 largest files in /kaggle/working
working = Path("/kaggle/working")
if working.exists():
    files = sorted(
        working.rglob("*"),
        key=lambda f: f.stat().st_size if f.is_file() else 0,
        reverse=True,
    )
    big = [(f, f.stat().st_size) for f in files if f.is_file()][:10]
    if big:
        print()
        print("  Largest files in /kaggle/working:")
        for f, sz in big:
            print(f"    {sz/1e9:.2f} GB  {f.relative_to(working)}")

print()
print("=" * 60)
print("GPU TIER DETECTION")
print("=" * 60)
try:
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        per_gpu = props.total_memory / 1024**3
        total   = per_gpu * n
        arch_map = {9: "Hopper", 8: "Ampere", 7: "Turing", 6: "Pascal"}
        arch = arch_map.get(props.major, f"Unknown(sm_{props.major}x)")
        print(f"  GPUs          : {n}× {props.name}")
        print(f"  Per-GPU VRAM  : {per_gpu:.1f} GB")
        print(f"  Total VRAM    : {total:.1f} GB   ← backend_selector uses THIS")
        print(f"  Architecture  : {arch}  (sm_{props.major}{props.minor})")
        print()
        if arch == "Turing" and total >= 28:
            print("  ✅ Correct tier: Tier 2 — HF Transformers + AWQ (dual T4)")
        elif total > 70:
            print("  ✅ Correct tier: Tier 1 — vLLM BF16")
        elif total >= 28:
            print("  ✅ Correct tier: Tier 2 — HF Transformers + AWQ (non-Turing)")
        else:
            print(f"  ⚠️  Tier 3 — only {total:.1f} GB total, need ≥28 GB for Tier 2")
    else:
        print("  No CUDA available → Tier 4 (CPU/Llama.cpp)")
except ImportError:
    print("  torch not installed yet")

print()
print("=" * 60)
print("HOW TO FIX (if issues found above)")
print("=" * 60)
print("""
  1. Make kaggle_bootstrap.py Cell 1 of your notebook (before any imports).
  2. The fixed models/backend_selector.py already sets HF cache at import time.
  3. The fixed models/hf_backend.py uses max_memory + offload to prevent OOM.
  4. To clean up /kaggle/working right now, run:
       !find /kaggle/working -name "*.safetensors" -delete
       !find /kaggle/working -name "*.bin" -delete
       !find /kaggle/working -name "*.gguf" -delete
""")
