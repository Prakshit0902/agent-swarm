from __future__ import annotations
import asyncio, time, statistics, argparse
from models.backend_selector import choose
from models.inference import build_engine

PROMPTS = [
    "Write a Python function to compute Levenshtein distance.",
    "Explain Paged Attention in one paragraph.",
    "Refactor this function to async: def fetch(u): import requests; return requests.get(u).text",
    "Generate a pytest for `def add(a,b): return a+b`.",
]

async def main(n: int):
    choice, _ = choose()
    eng = build_engine(choice)
    # warm-up
    await eng.chat([{"role":"user","content":"hi"}], max_new_tokens=8)
    lats, toks = [], []
    for i in range(n):
        p = PROMPTS[i % len(PROMPTS)]
        t0 = time.time()
        out = await eng.chat([{"role":"user","content":p}], max_new_tokens=256)
        dt = time.time() - t0
        lats.append(dt); toks.append(len(out.split()) / dt)
        print(f"[{i+1}/{n}] {dt:.2f}s  ~{toks[-1]:.1f} tok/s")
    print("---")
    print(f"backend={choice.backend} model={choice.model_id} quant={choice.quantization}")
    print(f"latency p50={statistics.median(lats):.2f}s "
          f"p95={sorted(lats)[int(0.95*len(lats))-1]:.2f}s")
    print(f"throughput mean={statistics.mean(toks):.1f} tok/s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=8)
    asyncio.run(main(ap.parse_args().n))
