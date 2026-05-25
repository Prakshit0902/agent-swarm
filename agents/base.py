from __future__ import annotations
import json, re
from typing import Any
from models.inference import LLMEngine

JSON_RE = re.compile(r"\{[\s\S]*\}\s*$")

class Agent:
    def __init__(self, engine: LLMEngine, system: str, name: str):
        self.engine = engine; self.system = system; self.name = name

    async def run(self, user_payload: str, *, json_mode: bool = True,
                  history: list[dict] | None = None, **gen_kw) -> Any:
        msgs = [{"role":"system","content":self.system}]
        if history: msgs.extend(history)
        msgs.append({"role":"user","content":user_payload})
        raw = await self.engine.chat(msgs, **gen_kw)
        if not json_mode: return raw
        m = JSON_RE.search(raw.strip())
        if not m:
            # one repair retry
            repair = await self.engine.chat(msgs + [
                {"role":"assistant","content":raw},
                {"role":"user","content":"Re-output ONLY the JSON object."}
            ], **gen_kw)
            m = JSON_RE.search(repair.strip())
            raw = repair if m else raw
        try:
            return json.loads(m.group(0)) if m else {"raw": raw}
        except Exception:
            return {"raw": raw}
