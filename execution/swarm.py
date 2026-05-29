from __future__ import annotations
import asyncio, json, re, uuid
from rich.console import Console
from config.settings import settings
from models.backend_selector import choose
from models.inference import build_engine
from agents.planner import Planner
from agents.researcher import Researcher
from agents.repo_context import RepoContextAgent
from agents.architect import Architect
from agents.coder import Coder
from agents.reviewer import Reviewer
from agents.debugger import Debugger
from rag.store import RepoStore
from tools.patch import apply_patch
from tools.test_runner import run_pytest
from tools.linter import ruff
from memory.session import SessionMemory

DIFF_RE = re.compile(r"```diff\s*([\s\S]*?)```", re.MULTILINE)
console = Console()

class CodingSwarm:
    def __init__(self, repo_root=None, model_tier: int | str | None = None):
        choice, gpu = choose(force=model_tier)
        self.engine = build_engine(choice)
        self.planner   = Planner(self.engine)
        self.research  = Researcher(self.engine)
        self.repo_ctx  = RepoContextAgent(self.engine)
        self.arch      = Architect(self.engine)
        self.coder     = Coder(self.engine)
        self.review    = Reviewer(self.engine)
        self.debug     = Debugger(self.engine)
        self.store     = RepoStore()
        self.repo_root = repo_root or settings.workspace
        self.session   = SessionMemory(str(uuid.uuid4()))

    async def _parallel_context(self, task, plan):
        async def _research():
            if settings.parallel_research:
                return await self.research.investigate(task)
            return {}
        async def _retrieve():
            hits = self.store.query(task, k=8)
            return await self.repo_ctx.analyze(hits)
        async def _design():
            return await self.arch.run(f"TASK:{task}\nPLAN:{json.dumps(plan)[:4000]}")
        res, ctx, design = await asyncio.gather(_research(), _retrieve(), _design())
        return res, ctx, design

    def _apply_diffs(self, coder_output: str) -> list[str]:
        applied = []
        for m in DIFF_RE.finditer(coder_output):
            r = apply_patch(m.group(1))
            if r["ok"]: applied.append(m.group(1)[:120])
            else: console.print(f"[red]patch failed:[/red] {r.get('stderr')}")
        return applied

    async def solve(self, task: str) -> dict:
        console.rule(f"[bold cyan]Task:[/bold cyan] {task}")
        self.session.log("user","user",task)

        plan = await self.planner.run(task)
        self.session.log("assistant","planner",plan)
        console.print("[bold green]Plan[/bold green]:", plan)

        research, repo_ctx, design = await self._parallel_context(task, plan)
        self.session.log("assistant","researcher",research)
        self.session.log("assistant","repo_ctx",repo_ctx)
        self.session.log("assistant","architect",design)
        console.print("[bold blue]Research:[/bold blue]\n", research)
        console.print("[bold blue]Repo Context:[/bold blue]\n", repo_ctx)
        console.print("[bold blue]Design/Architecture:[/bold blue]\n", design)

        files_to_inspect = plan.get("files_to_inspect", []) if isinstance(plan, dict) else []

        for it in range(1, settings.max_iterations+1):
            console.rule(f"Iteration {it}")
            
            full_files_content = ""
            for fpath in files_to_inspect:
                try:
                    full_path = settings.workspace / fpath
                    if full_path.exists() and full_path.is_file():
                        content = full_path.read_text("utf-8", "replace")
                        full_files_content += f"--- {fpath} ---\n{content}\n\n"
                except Exception as e:
                    full_files_content += f"--- {fpath} ---\nError reading file: {e}\n\n"

            code_out = await self.coder.code(plan, design, repo_ctx, full_files=full_files_content)
            self.session.log("assistant","coder",code_out)
            console.print("[bold magenta]Coder Output:[/bold magenta]\n", code_out)
            self._apply_diffs(code_out)

            lint = ruff(".")
            tests = run_pytest(".", "-q")
            exec_report = {"lint": lint, "tests": tests}
            self.session.log("tool","executor",exec_report)

            review = await self.review.run(json.dumps(exec_report)[:8000])
            self.session.log("assistant","reviewer",review)
            console.print("[yellow]Review:[/yellow]", review)

            if isinstance(review, dict) and review.get("verdict") == "pass":
                console.print("[bold green]✔ All checks passed[/bold green]")
                return {"status":"success","iterations":it,"review":review}

            dbg = await self.debug.run(json.dumps({"review":review,"exec":exec_report})[:8000])
            self.session.log("assistant","debugger",dbg)
            console.print("[bold red]Debugger:[/bold red]\n", dbg)
            
            if isinstance(dbg, dict):
                for f in dbg.get("files_to_edit", []):
                    if f not in files_to_inspect:
                        files_to_inspect.append(f)

            # feed debug result back as design refinement
            design = {"design": design, "patch_plan": dbg}

        return {"status":"max_iterations","review":review}
