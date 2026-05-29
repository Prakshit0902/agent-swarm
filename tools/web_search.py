from __future__ import annotations
from .registry import tool

@tool("web_search","Search the web via DuckDuckGo and fetch snippets.")
async def web_search(query: str, k: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []
    out = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with DDGS() as d:
            for r in d.text(query, max_results=k):
                out.append({"title": r.get("title",""), "url": r.get("href",""),
                            "snippet": r.get("body","")})
    return out
