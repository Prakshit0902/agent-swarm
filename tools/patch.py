from __future__ import annotations
import subprocess, tempfile, os
from config.settings import settings
from .registry import tool

def _fuzzy_apply_single_file(lines: list[str]) -> dict:
    target_file = None
    for line in lines:
        if line.startswith("+++"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                p = parts[1].strip().strip('"\'')
                if p.startswith("b/"):
                    p = p[2:]
                target_file = p
                break
        elif line.startswith("---") and not target_file:
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                p = parts[1].strip().strip('"\'')
                if p.startswith("a/"):
                    p = p[2:]
                target_file = p

    if not target_file:
        return {"ok": False, "stderr": "Could not parse target file from diff"}

    file_path = settings.workspace / target_file
    if not file_path.exists():
        file_content = ""
    else:
        file_content = file_path.read_text("utf-8", "replace")

    hunks = []
    current_hunk = []
    for line in lines:
        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
                current_hunk = []
        elif line.startswith("---") or line.startswith("+++"):
            continue
        else:
            if target_file:
                current_hunk.append(line)
    if current_hunk:
        hunks.append(current_hunk)

    if not hunks:
        hunks = [[l for l in lines if not (l.startswith("---") or l.startswith("+++"))]]

    applied_any = False
    new_content = file_content

    for hunk in hunks:
        search_lines = []
        replace_lines = []
        for line in hunk:
            if not line:
                search_lines.append("")
                replace_lines.append("")
                continue
            if line.startswith("-"):
                search_lines.append(line[1:])
            elif line.startswith("+"):
                replace_lines.append(line[1:])
            else:
                val = line[1:] if line.startswith(" ") else line
                search_lines.append(val)
                replace_lines.append(val)

        search_str = "\n".join(search_lines)
        replace_str = "\n".join(replace_lines)

        if search_str in new_content and search_str.strip():
            new_content = new_content.replace(search_str, replace_str, 1)
            applied_any = True
        elif not search_str.strip():
            if new_content and not new_content.endswith("\n"):
                new_content += "\n"
            new_content += replace_str
            if not new_content.endswith("\n"):
                new_content += "\n"
            applied_any = True
        else:
            norm_search = "\n".join(l.strip() for l in search_lines if l.strip())
            if not norm_search:
                if new_content and not new_content.endswith("\n"):
                    new_content += "\n"
                new_content += replace_str
                if not new_content.endswith("\n"):
                    new_content += "\n"
                applied_any = True
                continue

            file_lines = new_content.splitlines()
            search_line_list = [l.strip() for l in search_lines if l.strip()]
            
            match_index = -1
            for idx in range(len(file_lines) - len(search_line_list) + 1):
                window = [l.strip() for l in file_lines[idx:idx+len(search_line_list)]]
                if window == search_line_list:
                    match_index = idx
                    break
            
            if match_index != -1:
                before = file_lines[:match_index]
                after = file_lines[match_index + len(search_line_list):]
                new_content = "\n".join(before) + "\n" + replace_str + "\n" + "\n".join(after)
                applied_any = True

    if applied_any:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_content, "utf-8")
        return {"ok": True, "stdout": f"Fuzzy applied successfully to {target_file}"}
    else:
        return {"ok": False, "stderr": f"Fuzzy matching failed: context block not found in {target_file}"}

def fuzzy_apply_diff(diff_text: str) -> dict:
    file_diffs = []
    current_diff = []
    
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            if current_diff:
                file_diffs.append(current_diff)
            current_diff = [line]
        elif current_diff:
            current_diff.append(line)
        elif line.startswith("+++ ") and not current_diff:
            current_diff = [line]
            
    if current_diff:
        file_diffs.append(current_diff)
    
    if not file_diffs:
        file_diffs = [diff_text.splitlines()]

    all_applied = True
    messages = []
    
    for diff_lines in file_diffs:
        res = _fuzzy_apply_single_file(diff_lines)
        if not res["ok"]:
            all_applied = False
            messages.append(res["stderr"])
        else:
            messages.append(res["stdout"])

    return {"ok": all_applied, "stdout": "\n".join(messages), "stderr": "\n".join(messages)}

@tool("apply_patch","Apply a unified diff to the workspace.")
def apply_patch(diff_text: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(diff_text); path = f.name
    try:
        r = subprocess.run(["git","apply","--whitespace=nowarn",path],
                           cwd=settings.workspace, capture_output=True, text=True)
        if r.returncode != 0:
            r2 = subprocess.run(["patch","-p1","-i",path],
                                cwd=settings.workspace, capture_output=True, text=True)
            if r2.returncode != 0:
                # Fallback to fuzzy matching
                return fuzzy_apply_diff(diff_text)
            return {"ok": r2.returncode == 0,
                    "stdout": r2.stdout, "stderr": r2.stderr}
        return {"ok": True, "stdout": r.stdout, "stderr": r.stderr}
    finally:
        os.unlink(path)
