"""Step 3 of the review: three specialist reviewers write candidate comments.

  correctness  - logic, edge cases, error handling, performance
  security     - injection, secrets, unsafe deserialization, auth, crypto
  design       - over-engineering, duplication, fit with the request and the project, tests

With an LLM each specialist sees the numbered diff, the intent check and the evidence relevant
to it, and is run several times (the verifier keeps what most runs agree on). Without an LLM
each specialist turns its evidence into concrete, line-anchored comments.
"""

from __future__ import annotations

import concurrent.futures as cf
import re

from agent import intent as intent_step
from agent import llm
from agent.state import Comment, Evidence, PRInput
from risk.features import GEN_RE, LOCK_RE

SPECIALISTS = {
    "correctness": {
        "cats": {"correctness", "performance"},
        "focus": "logic errors, wrong conditions, off-by-one, unhandled None/null/empty input, error handling "
        "that hides failures, race conditions, resource leaks, broken API contracts, and real "
        "performance problems (N+1 queries, quadratic loops on unbounded input, blocking calls)",
    },
    "security": {
        "cats": {"security"},
        "focus": "injection (SQL/shell/template), XSS, hard-coded secrets, unsafe deserialization, path traversal, "
        "missing authentication/authorization checks, weak crypto or randomness, disabled TLS checks, "
        "sensitive data in logs",
    },
    "design": {
        "cats": {"design", "tests", "intent"},
        "focus": "over-engineering (abstractions, options or files nobody asked for), duplicated code, changes that do "
        "not fit the project's existing patterns, code that does not serve the issue, dead code, and "
        "missing tests for new behaviour",
    },
}

SUGGEST = {
    "sec.eval": "Parse the input explicitly (e.g. `json.loads`, `ast.literal_eval`) instead of evaluating it.",
    "sec.exec": "Replace dynamic execution with an explicit dispatch table.",
    "sec.shell": "Pass an argument list and keep `shell=False`: `subprocess.run([cmd, arg], check=True)`.",
    "sec.os_system": "Use `subprocess.run([...], check=True)` with an argument list.",
    "sec.pickle": "Use JSON (or a schema-validated format) for data that crosses a trust boundary.",
    "sec.yaml": "Use `yaml.safe_load(...)`.",
    "sec.sql_concat": 'Use placeholders: `cursor.execute("... WHERE id = %s", (user_id,))`.',
    "sec.sql_fstring": 'Use placeholders: `cursor.execute("... WHERE id = %s", (user_id,))`.',
    "sec.secret": "Read it from an environment variable or secret manager and rotate the leaked value.",
    "sec.aws_key": "Revoke this key now and load credentials from the environment.",
    "sec.tls_off": "Keep certificate verification on; point to a CA bundle if a private CA is needed.",
    "sec.weak_hash": "Use `hashlib.sha256` for integrity, and bcrypt/argon2 for passwords.",
    "sec.random_token": "Use `secrets.token_urlsafe()` / `secrets.randbelow()`.",
    "sec.innerhtml": "Use `textContent` or a sanitizer such as DOMPurify.",
    "sec.jwt_none": 'Pin the expected algorithm, e.g. `algorithms=["HS256"]`, and always verify signatures.',
    "cor.bare_except": "Catch the specific exception you expect (e.g. `except ValueError:`) and log or re-raise.",
    "cor.except_pass": "Log the error or handle it; silently passing hides failures.",
    "cor.mutable_default": "Default to `None` and create the list/dict inside the function.",
    "cor.no_timeout": "Add `timeout=10` (or a configured value) to the call.",
    "cor.open_no_with": "Use `with open(...) as fh:`.",
    "cor.empty_catch": "Handle or at least log the error in the catch block.",
    "des.debug_print": "Remove it or use the project's logger at debug level.",
    "des.todo": "Resolve it in this PR or open a tracked issue and link it.",
    "duplicated-block": "Extract the shared lines into one helper and call it from both places.",
    "no-tests": "Add a test that exercises the new behaviour, including one failure/edge case.",
    "unused-def": "Remove it, or wire it up where it is needed.",
    "cyclomatic": "Split the function into smaller helpers or use early returns to flatten branches.",
}

TITLES = {
    "security": "Security",
    "correctness": "Correctness",
    "performance": "Performance",
    "design": "Design",
    "tests": "Tests",
    "intent": "Scope",
}


def _short(msg: str, n: int = 90) -> str:
    s = msg.split(";")[0].split(". ")[0].strip().rstrip(".")
    return s if len(s) <= n else s[: n - 1] + "…"


DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|func|fn)\s+\w"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"
    r"|^\s*@\w+(?:\.\w+)*\("  # a decorated route/handler, e.g. @app.route("/search")
)


FILE_LEVEL_RULES = {"no-tests"}  # findings about a whole file; the collector puts them on its first line


def default_line(f) -> int | None:
    """Where to anchor a file-level finding: the largest new definition, else the first change.

    "No tests for this change" reads best on the main function it is about, not on an import at
    the top of the file. A definition's size runs to the next new definition (or the last added line).
    """
    added = f.added_lines()
    if not added:
        return None
    defs = [no for no, text in added if DEF_RE.match(text) and not text.lstrip().startswith("@")]
    if not defs:
        return added[0][0]
    ends = defs[1:] + [added[-1][0] + 1]
    return max(zip(defs, ends), key=lambda d: (d[1] - d[0], -d[0]))[0]


def requirement_comments(pr: PRInput, intent: dict, limit: int = 3) -> list[Comment]:
    """Turn requirements the intent checker marked missing/partial into line-anchored comments.

    Requirements about tests are skipped (the evidence collector's "no-tests" finding covers them).
    A requirement is anchored to the changed line that shares the most words with it; one with no
    related changed line is left to the review summary rather than pinned to an arbitrary line.
    Requirements that land within 3 lines of each other are folded into one comment.
    """
    groups: list[dict] = []
    for r in intent.get("requirements", []):
        if r.get("status") not in ("missing", "partial") or intent_step.is_test_requirement(r.get("text", "")):
            continue
        at = intent_step.best_line(pr, r["text"], r.get("files"))
        if at is None:
            continue
        g = next((g for g in groups if g["file"] == at[0] and abs(g["line"] - at[1]) <= 3), None)
        if g is None:
            g = {"file": at[0], "line": at[1], "reqs": []}
            groups.append(g)
        g["reqs"].append(r)
    out = []
    for g in groups[:limit]:
        reqs = g["reqs"]
        lines = []
        for r in reqs:
            why = r.get("evidence") or (
                "none of " + ", ".join(f"`{t}`" for t in r["missing_terms"][:4]) + " appear in the change"
                if r.get("missing_terms")
                else ""
            )
            lines.append(f"- [{r['status']}] {r['text']}" + (f" ({why})" if why else ""))
        one = len(reqs) == 1
        out.append(
            Comment(
                file=g["file"],
                line=g["line"],
                category="intent",
                severity="medium" if any(r["status"] == "missing" for r in reqs) else "low",
                title="Scope: requirement not fully met: " + _short(reqs[0]["text"], 70)
                if one
                else "Scope: requirements from the issue not fully met",
                body=("The issue asks for something" if one else "The issue asks for things")
                + " this change does not clearly do:\n"
                + "\n".join(lines),
                source="design/tools",
                evidence=[f"intent:{r['id']}" for r in reqs],
                suggestion="Implement it here (and cover it with a test), or say in the PR why it is out of scope.",
            )
        )
    return out


def tool_comments(name: str, pr: PRInput, evidence: list[Evidence], intent: dict) -> list[Comment]:
    cats = SPECIALISTS[name]["cats"]
    out = []
    for e in evidence:
        if e.category not in cats:
            continue
        f = pr.file(e.file)
        line = e.line
        quote = e.snippet
        if f is not None and (line is None or e.rule in FILE_LEVEL_RULES):
            moved = default_line(f)
            if moved is not None and moved != line:
                line, quote = moved, ""  # the collector's snippet belongs to the old line
        sug = SUGGEST.get(e.rule) or SUGGEST.get(e.rule.split(".")[-1]) or ""
        body = e.message + (f" Found by {e.tool} ({e.rule})." if e.tool != "rules" else "")
        out.append(
            Comment(
                file=e.file,
                line=line,
                category=e.category,
                severity=e.severity,
                title=f"{TITLES.get(e.category, 'Review')}: {_short(e.message)}",
                body=body,
                source=f"{name}/tools",
                evidence=[e.id],
                quote=quote,
                suggestion=sug,
            )
        )
    if name == "design":
        for u in intent.get("unrequested", [])[:4]:
            f = pr.file(u["file"])
            if f is None or not f.changed_lines() or LOCK_RE.search(f.path) or GEN_RE.search(f.path):
                continue
            out.append(
                Comment(
                    file=f.path,
                    line=min(f.changed_lines()),
                    category="intent",
                    severity="low",
                    title="Scope: change not covered by the request",
                    body=f"This file changes {f.additions + f.deletions} lines but none of it maps to the "
                    f"issue/description ({u['reason']}). If it is needed, say why in the PR; otherwise "
                    f"move it to a separate PR.",
                    source="design/tools",
                    evidence=[f"intent:{f.path}"],
                    suggestion="Split unrelated changes into their own PR.",
                )
            )
        out.extend(requirement_comments(pr, intent))
    return out


SYSTEM = """You are the {name} specialist in a code review agent. Focus ONLY on: {focus}.
Rules:
- Comment only on lines that are ADDED (marked +) in the diff. Use the new-file line numbers shown at the left.
- Each comment must quote the exact code line it is about in "quote".
- If a comment is backed by a tool finding, list its evidence id (E#) in "evidence".
- Prefer few, high-value comments. No style nits, no praise, no restating the diff. Maximum 6 comments.
- If nothing in your focus is wrong, return {{"comments": []}}.
Reply with JSON only:
{{"comments":[{{"file":"<path>","line":<int>,"severity":"critical|medium|low","title":"<short>",
"body":"<what is wrong and why it matters>","suggestion":"<concrete fix>",
"quote":"<exact line>","evidence":["E1"]}}]}}"""


def _prompt(name, pr: PRInput, evidence, intent) -> str:
    cats = SPECIALISTS[name]["cats"]
    ev = [e for e in evidence if e.category in cats] or evidence[:15]
    ev_txt = "\n".join(f"{e.id} {e.file}:{e.line or '-'} [{e.tool}/{e.rule}] {e.message}" for e in ev[:40]) or "(none)"
    req = "\n".join(f"- [{r['status']}] {r['text']}" for r in intent.get("requirements", [])) or "(none)"
    unreq = "\n".join(f"- {u['file']}: {u['reason']}" for u in intent.get("unrequested", [])) or "(none)"
    files = [f for f in pr.files if not f.binary and not LOCK_RE.search(f.path) and not GEN_RE.search(f.path)]
    budget, parts = 16000, []
    for f in sorted(files, key=lambda x: -(x.additions)):
        block = f"### {f.path} ({f.status})\n{f.numbered_patch(250)}\n"
        if budget - len(block) < 0:
            parts.append(f"### {f.path} (omitted for length, +{f.additions}/-{f.deletions})")
            continue
        budget -= len(block)
        parts.append(block)
    return (
        f"PR: {pr.title}\n{pr.description[:1500]}\n\nREQUIREMENTS CHECK:\n{req}\nUNREQUESTED CHANGES:\n{unreq}\n\n"
        f"TOOL EVIDENCE:\n{ev_txt}\n\nDIFF:\n" + "\n".join(parts)
    )


def llm_review(name: str, pr: PRInput, evidence, intent, run: int, temperature: float) -> list[Comment]:
    sysmsg = SYSTEM.format(name=name, focus=SPECIALISTS[name]["focus"])
    data, _ = llm.chat_json(
        sysmsg,
        _prompt(name, pr, evidence, intent),
        temperature=temperature,
        sample=run,
        tag=f"specialist/{name}",
        max_tokens=2048,
    )
    out = []
    for c in (data or {}).get("comments", [])[:8]:
        if not isinstance(c, dict) or not c.get("file"):
            continue
        try:
            line = int(c.get("line")) if c.get("line") not in (None, "") else None
        except (TypeError, ValueError):
            line = None
        sev = c.get("severity") if c.get("severity") in ("critical", "medium", "low") else "medium"
        out.append(
            Comment(
                file=str(c["file"]),
                line=line,
                category=name,
                severity=sev,
                title=str(c.get("title", ""))[:140] or f"{name} issue",
                body=str(c.get("body", ""))[:1500],
                source=f"{name}/llm",
                evidence=[str(x) for x in (c.get("evidence") or [])][:5],
                quote=str(c.get("quote", ""))[:300],
                run=run,
                suggestion=str(c.get("suggestion", ""))[:600],
            )
        )
    return out


def review(pr: PRInput, evidence, intent, use_llm: bool, runs: int = 3, names=None, progress=None) -> dict:
    names = names or list(SPECIALISTS)
    candidates: list[Comment] = []
    errors = []
    for n in names:  # tool evidence is always turned into candidates
        candidates.extend(tool_comments(n, pr, evidence, intent))
    if use_llm:
        jobs = [(n, r) for n in names for r in range(runs)]
        with cf.ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(llm_review, n, pr, evidence, intent, r, 0.2 if r == 0 else 0.7): (n, r) for n, r in jobs}
            for fut in cf.as_completed(futs):
                n, r = futs[fut]
                try:
                    candidates.extend(fut.result())
                except Exception as e:
                    errors.append(f"{n} run {r}: {type(e).__name__}: {str(e)[:120]}")
                if progress:
                    progress(f"{n} run {r + 1}/{runs} done")
    return {"candidates": candidates, "errors": errors, "runs": runs if use_llm else 1}