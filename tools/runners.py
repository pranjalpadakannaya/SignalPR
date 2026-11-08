"""Shared code-analysis tool runners (issue #10, task 1).

Runs Ruff (Python), ESLint (JavaScript), Semgrep (any language, when installed), Lizard
(cyclomatic complexity, when installed; built-in AST/heuristic counter otherwise) and the
built-in regex rules in `rules.py` on the CHANGED LINES of a pull request, and caches the
results on disk so the agent, the baselines and the risk-model features can all reuse them.

    from tools import ChangedFile, run_tools, files_from_diff
    files = files_from_diff(open("change.diff").read())        # or build ChangedFile yourself
    report = run_tools(files)                                   # -> ToolReport
    for f in report.findings: print(f.file, f.line, f.tool, f.rule, f.message)

Every Finding is on a changed line (or file-level, line=None), has a category
(security | correctness | design | performance | tests) and a severity (critical | medium | low).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from tools.diff import FileDiff, parse_unified_diff
from tools.lang import language_of
from tools.rules import run_rules

CACHE_DIR = Path(os.environ.get("SIGNALPR_TOOL_CACHE", Path.home() / ".cache" / "signalpr-tools"))
ALL_TOOLS = ("rules", "ruff", "eslint", "semgrep", "lizard", "complexity")

RUFF_SELECT = "E9,F,B,S,SIM,C901,BLE,PLE,PLW"
RUFF_IGNORE = "S101,E501,S311,S603,S607,S404"
ESLINT_RULES = {
    "no-eval": "error",
    "no-implied-eval": "error",
    "no-new-func": "error",
    "eqeqeq": "warn",
    "no-unused-vars": "warn",
    "complexity": ["warn", 12],
    "no-empty": "warn",
    "no-unreachable": "error",
    "no-dupe-keys": "error",
    "no-self-compare": "error",
    "no-constant-condition": "warn",
    "require-await": "warn",
    "no-await-in-loop": "warn",
    "no-fallthrough": "warn",
    "no-unsafe-finally": "error",
    "use-isnan": "error",
    "no-var": "warn",
}
CC_THRESHOLD = 10


@dataclass
class ChangedFile:
    path: str
    changed_lines: set[int]
    added_lines: list[tuple[int, str]]
    content: str | None = None  # full post-change content, if known (needed by ruff/eslint/lizard)
    status: str = "modified"

    @classmethod
    def from_filediff(cls, f: FileDiff) -> "ChangedFile":
        content = f.content
        if content is None and f.status == "added":
            content = "\n".join(t for _, t in f.added_lines())
        elif content is None and len(f.hunks) == 1 and f.hunks[0].new_start == 1:
            content = "\n".join(ln.text for ln in f.hunks[0].lines if ln.kind != "-")
        return cls(f.path, f.changed_lines(), f.added_lines(), content, f.status)


@dataclass
class Finding:
    file: str
    line: int | None
    tool: str
    rule: str
    category: str
    severity: str
    message: str

    def to_dict(self):
        return asdict(self)


@dataclass
class ToolReport:
    findings: list[Finding] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)  # tool -> "ran ..." / "not installed" / "cached" ...
    cache_hits: int = 0

    def by_file(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.file, []).append(f)
        return out

    def counts(self, path: str) -> dict[str, int]:
        """Per-file counts used as risk-model features."""
        fs = [f for f in self.findings if f.file == path]
        return {
            "tool_warnings": len(fs),
            "tool_security": sum(f.category == "security" for f in fs),
            "tool_critical": sum(f.severity == "critical" for f in fs),
            "tool_complexity": sum(f.rule in ("cyclomatic", "C901", "complexity") for f in fs),
        }


def files_from_diff(diff: str) -> list[ChangedFile]:
    return [ChangedFile.from_filediff(f) for f in parse_unified_diff(diff) if not f.binary and f.status != "removed"]


# ------------------------------------------------------------------ cache
@lru_cache(maxsize=None)
def tool_version(tool: str) -> str | None:
    exe = shutil.which(tool)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0][:60] or "unknown"
    except Exception:
        return "unknown"


def _key(tool: str, f: ChangedFile) -> str:
    h = hashlib.sha256()
    for part in (
        tool,
        tool_version(tool) or "builtin",
        f.path,
        f.content or "",
        json.dumps(sorted(f.changed_lines)),
        json.dumps(f.added_lines),
    ):
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()


def _cache_get(key: str):
    p = CACHE_DIR / key[:2] / f"{key}.json"
    try:
        return [Finding(**d) for d in json.loads(p.read_text())]
    except Exception:
        return None


def _cache_put(key: str, findings: list[Finding]):
    p = CACHE_DIR / key[:2] / f"{key}.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([x.to_dict() for x in findings]))
    except Exception:
        pass


# ------------------------------------------------------------------ individual tools (one file each)
def _ruff_meta(code: str) -> tuple[str, str]:
    if code.startswith("E9") or code in ("F821", "F822", "F823", "PLE0101"):
        return "correctness", "critical"
    if code == "S113":
        return "correctness", "medium"
    if code.startswith("SIM"):
        return "design", "low"
    if code.startswith("S"):
        sev = "critical" if code in ("S102", "S307", "S301", "S608", "S602", "S105", "S106", "S506") else "medium"
        return "security", sev
    if code.startswith(("B", "BLE", "PLW", "PLE")):
        return "correctness", "medium"
    if code == "C901":
        return "design", "medium"
    if code in ("F401", "F841", "F811"):
        return "design", "low"
    if code.startswith("F"):
        return "correctness", "medium"
    return "design", "low"


def _write(tmp: Path, f: ChangedFile) -> Path:
    p = tmp / f.path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f.content or "")
    return p


def run_ruff(f: ChangedFile, tmp: Path) -> list[Finding] | None:
    exe = shutil.which("ruff")
    if not exe or f.content is None or language_of(f.path) != "python":
        return None
    p = _write(tmp, f)
    out = subprocess.run(
        [
            exe,
            "check",
            "--isolated",
            "--output-format",
            "json",
            "--select",
            RUFF_SELECT,
            "--ignore",
            RUFF_IGNORE,
            "--exit-zero",
            str(p),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    res = []
    for d in json.loads(out.stdout or "[]"):
        cat, sev = _ruff_meta(d.get("code") or "")
        res.append(Finding(f.path, d["location"]["row"], "ruff", d.get("code") or "?", cat, sev, d.get("message", "")))
    return res


def run_eslint(f: ChangedFile, tmp: Path) -> list[Finding] | None:
    exe = shutil.which("eslint")
    if not exe or f.content is None or not f.path.endswith((".js", ".mjs", ".cjs", ".jsx")):
        return None
    p = _write(tmp, f)
    out = subprocess.run(
        [
            exe,
            "--no-config-lookup",
            "--parser-options",
            "ecmaVersion:latest",
            "--parser-options",
            "sourceType:module",
            "--format",
            "json",
            "--rule",
            json.dumps(ESLINT_RULES),
            str(p.relative_to(tmp)),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        cwd=tmp,
    )
    res = []
    for fobj in json.loads(out.stdout or "[]"):
        for m in fobj.get("messages", []):
            rid = m.get("ruleId")
            if not rid or m.get("fatal"):
                continue
            sec = rid in ("no-eval", "no-implied-eval", "no-new-func")
            cat = (
                "security"
                if sec
                else "design"
                if rid in ("complexity", "no-unused-vars", "no-var")
                else "performance"
                if rid == "no-await-in-loop"
                else "correctness"
            )
            sev = "critical" if sec else ("medium" if m.get("severity") == 2 or rid == "complexity" else "low")
            res.append(Finding(f.path, m.get("line"), "eslint", rid, cat, sev, m.get("message", "")))
    return res


def run_semgrep(f: ChangedFile, tmp: Path) -> list[Finding] | None:
    exe, cfg = shutil.which("semgrep"), os.environ.get("SIGNALPR_SEMGREP_CONFIG")
    if not exe or not cfg or f.content is None:
        return None
    p = _write(tmp, f)
    out = subprocess.run(
        [exe, "--config", cfg, "--json", "--quiet", "--metrics=off", str(p)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    res = []
    for r in json.loads(out.stdout or "{}").get("results", []):
        sev = {"ERROR": "critical", "WARNING": "medium"}.get(r.get("extra", {}).get("severity"), "low")
        res.append(
            Finding(
                f.path,
                r["start"]["line"],
                "semgrep",
                r["check_id"].split(".")[-1],
                "security",
                sev,
                r.get("extra", {}).get("message", "")[:300],
            )
        )
    return res


def run_lizard(f: ChangedFile, tmp: Path) -> list[Finding] | None:
    exe = shutil.which("lizard")
    if not exe or f.content is None or language_of(f.path) in ("none", "other", "markdown", "config", "docs"):
        return None
    p = _write(tmp, f)
    out = subprocess.run([exe, "--csv", str(p)], capture_output=True, text=True, timeout=60)
    res = []
    for row in out.stdout.splitlines():
        parts = row.split(",")
        if len(parts) < 11:
            continue
        try:
            ccn, start, end = int(parts[1]), int(parts[9]), int(parts[10])
        except ValueError:
            continue
        name = parts[7].strip('"')
        if ccn > CC_THRESHOLD and set(range(start, end + 1)) & f.changed_lines:
            res.append(
                Finding(
                    f.path,
                    start if start in f.changed_lines else min(set(range(start, end + 1)) & f.changed_lines),
                    "lizard",
                    "cyclomatic",
                    "design",
                    "medium" if ccn <= 20 else "critical",
                    f"`{name}` has cyclomatic complexity {ccn} (threshold {CC_THRESHOLD}).",
                )
            )
    return res


def python_complexity(content: str) -> dict[str, int]:
    """Per-function cyclomatic complexity for Python source (built-in, no dependencies)."""
    out = {}
    for node in ast.walk(ast.parse(content)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[f"{node.name}:{node.lineno}:{node.end_lineno or node.lineno}"] = 1 + sum(
                isinstance(
                    n,
                    (
                        ast.If,
                        ast.For,
                        ast.While,
                        ast.AsyncFor,
                        ast.ExceptHandler,
                        ast.With,
                        ast.IfExp,
                        ast.comprehension,
                        ast.Assert,
                    ),
                )
                or (isinstance(n, ast.BoolOp) and len(n.values) - 1)
                for n in ast.walk(node)
            )
    return out


def run_complexity(f: ChangedFile, tmp: Path) -> list[Finding]:
    """Built-in complexity check: Python AST when possible, branch-token count per hunk otherwise.
    Skipped for non-Python files when Lizard is installed (Lizard covers them)."""
    lang = language_of(f.path)
    res = []
    if lang == "python" and f.content:
        try:
            for key, cc in python_complexity(f.content).items():
                name, start, end = key.rsplit(":", 2)
                span = set(range(int(start), int(end) + 1)) & f.changed_lines
                if cc > CC_THRESHOLD and span:
                    res.append(
                        Finding(
                            f.path,
                            int(start) if int(start) in f.changed_lines else min(span),
                            "complexity",
                            "cyclomatic",
                            "design",
                            "medium" if cc <= 20 else "critical",
                            f"`{name}` has cyclomatic complexity {cc} (threshold {CC_THRESHOLD}).",
                        )
                    )
        except SyntaxError as e:
            if f.status == "added":
                res.append(
                    Finding(
                        f.path,
                        e.lineno,
                        "complexity",
                        "syntax-error",
                        "correctness",
                        "critical",
                        f"File does not parse: {e.msg}",
                    )
                )
        return res
    if shutil.which("lizard"):
        return res
    branches = sum(len(re.findall(r"\b(if|for|while|case|catch|else if)\b|&&|\|\|", t)) for _, t in f.added_lines)
    if branches > 14 and f.added_lines:
        res.append(
            Finding(
                f.path,
                f.added_lines[0][0],
                "complexity",
                "branchy-change",
                "design",
                "medium",
                f"This change adds {branches} branch points; the logic may be hard to follow and test.",
            )
        )
    return res


def run_builtin_rules(f: ChangedFile, tmp: Path) -> list[Finding]:
    from risk.features import DOC_RE, TEST_RE  # path classifiers shared with the risk model

    is_test, is_doc = bool(TEST_RE.search(f.path)), bool(DOC_RE.search(f.path))
    res = []
    for rule, no, _ in run_rules(language_of(f.path), f.added_lines):
        if rule.id == "des.debug_print" and (is_test or is_doc):
            continue
        if rule.id == "cor.assert_runtime" and is_test:
            continue
        if rule.category == "security" and is_test and rule.severity != "critical":
            continue
        res.append(Finding(f.path, no, "rules", rule.id, rule.category, rule.severity, rule.message))
    return res


RUNNERS = {
    "rules": run_builtin_rules,
    "ruff": run_ruff,
    "eslint": run_eslint,
    "semgrep": run_semgrep,
    "lizard": run_lizard,
    "complexity": run_complexity,
}


# ------------------------------------------------------------------ public entry point
def run_tools(files: list[ChangedFile], tools: tuple[str, ...] = ALL_TOOLS, use_cache: bool = True) -> ToolReport:
    """Run every requested tool on every changed file; keep findings on changed lines only."""
    report = ToolReport()
    ran: dict[str, int] = {t: 0 for t in tools}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for f in files:
            for t in tools:
                key = _key(t, f)
                found = _cache_get(key) if use_cache else None
                if found is not None:
                    report.cache_hits += 1
                else:
                    try:
                        found = RUNNERS[t](f, tmp)
                    except Exception as e:  # a broken tool must not break the review
                        report.status[t] = f"error: {type(e).__name__}: {str(e)[:120]}"
                        found = []
                    if found is None:  # tool not applicable / not installed
                        continue
                    if use_cache:
                        _cache_put(key, found)
                ran[t] += 1
                for x in found:
                    if x.line is None or not f.changed_lines or x.line in f.changed_lines:
                        if not any((y.file, y.line, y.rule) == (x.file, x.line, x.rule) for y in report.findings):
                            report.findings.append(x)
    for t in tools:
        if t in report.status:
            continue
        exe_tool = t not in ("rules", "complexity")
        if exe_tool and not shutil.which(t):
            report.status[t] = "not installed"
        elif t == "semgrep" and not os.environ.get("SIGNALPR_SEMGREP_CONFIG"):
            report.status[t] = "installed; set SIGNALPR_SEMGREP_CONFIG to enable"
        else:
            report.status[t] = f"ran on {ran[t]} file(s)"
    return report
