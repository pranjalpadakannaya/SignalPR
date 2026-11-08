"""Step 2 of the review: evidence collector.

Gathers facts the reviewers' comments must point to, on the changed code only:
  * Ruff, ESLint, Semgrep, Lizard and the built-in rules/complexity counter (tools.run_tools, cached)
  * blocks of added code duplicated within the PR
  * definitions added or changed in the PR and where they are used (other changed files and,
    with a local checkout, the rest of the repository); unused new abstractions are flagged
  * the project's tests, when a local checkout and test command are given, plus a
    "code changed but no tests" check
Every finding becomes an Evidence item with an id (E1, E2, ...).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from agent.state import Evidence, PRInput
from risk.features import DOC_RE, GEN_RE, LOCK_RE, TEST_RE
from tools.lang import CODE_LANGS, language_of
from tools.runners import ALL_TOOLS, ChangedFile, run_tools

DEF_PATTERNS = {
    "python": re.compile(r"\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\("),
    "javascript": re.compile(
        r"\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"
        r"|\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("
    ),
    "java": re.compile(r"\s*(?:public|private|protected|static|\s)*[\w<>\[\]]+\s+([a-z]\w*)\s*\([^;]*$"),
    "go": re.compile(r"\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("),
}
DEF_PATTERNS["typescript"] = DEF_PATTERNS["javascript"]
DEF_PATTERNS["kotlin"] = DEF_PATTERNS["csharp"] = DEF_PATTERNS["java"]
CLASS_RE = re.compile(r"\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_]\w*)")


class Collector:
    def __init__(self, pr: PRInput, run_tests: bool = True):
        self.pr = pr
        self.items: list[Evidence] = []
        self.tools: dict[str, str] = {}
        self.run_tests = run_tests

    def add(self, file, line, tool, rule, category, severity, message, snippet=""):
        f = self.pr.file(file)
        if f is not None and line is not None and f.changed_lines() and line not in f.changed_lines():
            return  # only changed lines (or file-level findings with line=None)
        if any((e.file, e.line, e.rule) == (file, line, rule) for e in self.items):
            return
        text = snippet or (f.new_line_text(line) if f is not None and line else "") or ""
        self.items.append(
            Evidence(f"E{len(self.items) + 1}", file, line, tool, rule, category, severity, message, text.strip()[:200])
        )

    def content(self, f) -> str | None:
        if f.content is not None:
            return f.content
        if self.pr.repo_path:
            p = Path(self.pr.repo_path) / f.path
            if p.is_file() and p.stat().st_size < 600_000:
                try:
                    return p.read_text(errors="replace")
                except OSError:
                    return None
        return None

    # ------------------------------------------------------------ static analysis
    def static_tools(self, tools=ALL_TOOLS, use_cache: bool = True):
        files = []
        for f in self.pr.files:
            if f.binary or f.status == "removed" or LOCK_RE.search(f.path) or GEN_RE.search(f.path):
                continue
            cf = ChangedFile.from_filediff(f)
            cf.content = cf.content if cf.content is not None else self.content(f)
            files.append(cf)
        report = run_tools(files, tools=tools, use_cache=use_cache)
        for x in report.findings:
            self.add(x.file, x.line, x.tool, x.rule, x.category, x.severity, x.message)
        self.tools.update(report.status)
        return report

    # ------------------------------------------------------------ duplication
    def duplication(self):
        seen = defaultdict(list)
        for f in self.pr.files:
            if TEST_RE.search(f.path) or GEN_RE.search(f.path):
                continue
            added = [(n, re.sub(r"\s+", " ", t.strip())) for n, t in f.added_lines()]
            added = [(n, t) for n, t in added if len(t) > 3 and not t.startswith(("#", "//", "import", "from "))]
            for i in range(len(added) - 3):
                block = "\n".join(t for _, t in added[i : i + 4])
                if len(block) >= 90:
                    seen[hashlib.md5(block.encode()).hexdigest()].append((f.path, added[i][0]))
        reported = set()
        for locs in seen.values():
            if len(locs) < 2:
                continue
            (f0, l0), (f1, l1) = locs[0], locs[1]
            if (f1, l1 // 10) in reported or (f0 == f1 and abs(l0 - l1) < 4):
                continue
            reported.add((f1, l1 // 10))
            self.add(
                f1,
                l1,
                "duplication",
                "duplicated-block",
                "design",
                "medium",
                f"These lines duplicate {f0}:{l0} in the same PR; extract a shared helper.",
            )
            if len(reported) >= 6:
                break
        self.tools["duplication"] = f"{len(reported)} duplicated block(s)"

    # ------------------------------------------------------------ definitions and usage
    def usage(self) -> dict:
        defs = []
        for f in self.pr.files:
            pat = DEF_PATTERNS.get(language_of(f.path))
            for no, t in f.added_lines():
                cm = CLASS_RE.match(t)
                if cm:
                    defs.append((f, no, cm.group(1), "class"))
                    continue
                m = pat.match(t) if pat else None
                if m:
                    name = next(g for g in m.groups() if g)
                    if not (name.startswith("__") or name.startswith("test") or name in ("main", "setUp", "tearDown")):
                        defs.append((f, no, name, "def"))
        if not defs:
            return {}
        corpus = {f.path: (self.content(f) or "\n".join(t for _, t in f.added_lines())) for f in self.pr.files}
        out = {}
        for f, no, name, kind in defs[:40]:
            word = re.compile(rf"\b{re.escape(name)}\b")
            refs = [p for p, txt in corpus.items() if len(word.findall(txt)) > (1 if p == f.path else 0)]
            repo_refs = self._grep_repo(name, f.path)
            out[name] = {
                "defined_in": f"{f.path}:{no}",
                "kind": kind,
                "used_in": sorted(set(refs) | set(repo_refs))[:10],
            }
            if refs or repo_refs:
                continue
            if kind == "class":
                sub = re.compile(rf"\(\s*{re.escape(name)}\s*\)|extends\s+{re.escape(name)}\b")
                if not any(sub.search(txt) for txt in corpus.values()):
                    self.add(
                        f.path,
                        no,
                        "usage",
                        "unused-abstraction",
                        "design",
                        "medium",
                        f"New class `{name}` is not used anywhere in this PR; the abstraction may not be needed.",
                    )
            elif name.startswith("_"):
                self.add(
                    f.path,
                    no,
                    "usage",
                    "unused-def",
                    "design",
                    "low",
                    f"Private helper `{name}` is added but never referenced.",
                )
        self.tools["usage"] = f"{len(out)} new definition(s) checked"
        return out

    def _grep_repo(self, name: str, own: str) -> list[str]:
        if not (self.pr.repo_path and shutil.which("grep")):
            return []
        try:
            r = subprocess.run(
                ["grep", "-rIlw", "--exclude-dir=.git", "--exclude-dir=node_modules", name, self.pr.repo_path],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        rel = [os.path.relpath(x, self.pr.repo_path) for x in r.stdout.split()]
        return [x for x in rel if x != own][:10]

    # ------------------------------------------------------------ tests
    def tests(self):
        code = [
            f
            for f in self.pr.files
            if language_of(f.path) in CODE_LANGS
            and not TEST_RE.search(f.path)
            and not GEN_RE.search(f.path)
            and not DOC_RE.search(f.path)
            and f.additions >= 5
        ]
        has_tests = any(TEST_RE.search(f.path) for f in self.pr.files)
        added = sum(f.additions for f in code)
        if code and not has_tests and added >= 20:
            f = max(code, key=lambda x: x.additions)
            first = min(f.changed_lines()) if f.changed_lines() else None
            self.add(
                f.path,
                first,
                "tests",
                "no-tests",
                "tests",
                "medium",
                f"{added} lines of code changed in {len(code)} file(s) but no test files were added or updated.",
            )
        if not (self.run_tests and self.pr.repo_path and self.pr.test_cmd):
            self.tools["tests"] = "not run (needs a local checkout and a test command)"
            return
        try:
            r = subprocess.run(
                self.pr.test_cmd,
                shell=True,
                cwd=self.pr.repo_path,  # noqa: S602 - the user's own command
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            self.tools["tests"] = f"error: {e}"
            return
        tail = (r.stdout + r.stderr)[-1500:]
        self.tools["tests"] = f"exit code {r.returncode}"
        if r.returncode != 0:
            failing = re.findall(r"FAILED\s+(\S+)", tail)[:5]
            target = next((t.split("::")[0] for t in failing if self.pr.file(t.split("::")[0])), None)
            self.add(
                target or self.pr.files[0].path,
                None,
                "tests",
                "tests-fail",
                "correctness",
                "critical",
                "The test suite fails on this branch" + (f": {', '.join(failing)}" if failing else "") + ".",
                tail[-300:],
            )


def collect(pr: PRInput, run_tests: bool = True, tools=ALL_TOOLS, use_cache: bool = True) -> dict:
    """Run every evidence source. Returns {"items": [Evidence], "tools": {name: status}, "usage": {...}}."""
    c = Collector(pr, run_tests=run_tests)
    c.static_tools(tools, use_cache)
    c.duplication()
    usage = c.usage()
    c.tests()
    return {"items": c.items, "tools": c.tools, "usage": usage}
