import asyncio, time
from execution.swarm import CodingSwarm

TASKS = [
  "Create utils.py with `slugify(s)` and a pytest in tests/test_slugify.py.",
  "Add type hints to all functions in utils.py and pass ruff.",
  "Implement an LRU cache decorator in cache.py with tests.",
]

async def main():
    sw = CodingSwarm()
    for t in TASKS:
        t0=time.time(); r=await sw.solve(t); print(t, round(time.time()-t0,1),"s", r["status"])

if __name__ == "__main__":
    asyncio.run(main())
