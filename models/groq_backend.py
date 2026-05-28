from __future__ import annotations
import os, json
import httpx
from typing import AsyncIterator
from .backend_selector import BackendChoice
from config.settings import settings

class GroqEngine:
    def __init__(self, choice: BackendChoice):
        self.choice = choice
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.model = os.environ.get("GROQ_MODEL", choice.model_id)
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please obtain a key from Groq and set it in your environment."
            )

    async def chat(self, messages: list[dict], **kw) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kw.get("temperature", settings.temperature),
            "max_tokens": kw.get("max_new_tokens", settings.max_new_tokens),
            "top_p": settings.top_p,
        }
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90.0
            )
            r.raise_for_status()
            res = r.json()
            return res["choices"][0]["message"]["content"]

    async def stream(self, messages: list[dict], **kw) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kw.get("temperature", settings.temperature),
            "max_tokens": kw.get("max_new_tokens", settings.max_new_tokens),
            "top_p": settings.top_p,
            "stream": True,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90.0
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            pass
