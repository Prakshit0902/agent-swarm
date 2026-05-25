from __future__ import annotations
from pathlib import Path
from tree_sitter_languages import get_parser
LANG_BY_EXT = {".py":"python",".js":"javascript",".ts":"typescript",".go":"go",
               ".rs":"rust",".java":"java",".cpp":"cpp",".c":"c"}

def chunk_text(text: str, target: int = 1600, overlap: int = 200) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+target]); i += target - overlap
    return out

def chunk_file(path: Path) -> list[dict]:
    text = path.read_text("utf-8","replace")
    lang = LANG_BY_EXT.get(path.suffix)
    chunks = []
    if lang:
        try:
            parser = get_parser(lang); tree = parser.parse(text.encode("utf-8"))
            for node in tree.root_node.children:
                seg = text[node.start_byte:node.end_byte]
                if len(seg) < 40: continue
                for c in chunk_text(seg):
                    chunks.append({"text": c, "path": str(path), "lang": lang,
                                   "start": node.start_point[0]})
            if chunks: return chunks
        except Exception:
            pass
    return [{"text": c, "path": str(path), "lang": lang or "text", "start": 0}
            for c in chunk_text(text)]
