"""Step 1 of the review: intent checker.

Turns the issue + PR description into a checklist and marks what the code covers, what it
misses, and which changed files nobody asked for. Uses the LLM when available; otherwise a
lexical matcher (identifier-aware token overlap between each requirement and each file's diff).
"""

from __future__ import annotations

import re

from agent import llm
from agent.state import PRInput
from risk.features import DOC_RE, LOCK_RE, TEST_RE

STOP = set(
    """a an the and or but if then else of to in on for with without by from at as is are was were be been
being this that these those it its we you i they he she them our your their there here should must shall will would can
could may might do does did done not no yes when where which who what how why all any each every some such into onto
also only just more most less least than too very via per etc e g ie use using used make made add added adds new
please need needs needed want wants able support supports issue bug pr pull request code file files change changes
fix fixes fixed implement implements implementation feature currently expected behavior behaviour
like so get set""".split()
)

ACTION = re.compile(
    r"\b(should|must|need|needs|add|support|allow|implement|ensure|return|handle|fix|create|remove|"
    r"update|validate|expose|accept|reject|raise|throw|log|display|show|prevent|avoid|when|if)\b",
    re.I,
)


def _tokens(text: str) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    out = set()
    for t in re.findall(r"[A-Za-z][A-Za-z0-9]+", text.replace("_", " ").replace("-", " ")):
        t = t.lower()
        if len(t) > 2 and t not in STOP:
            out.add(t[:-1] if t.endswith("s") and len(t) > 4 else t)
    return out


def extract_requirements(pr: PRInput, limit: int = 8) -> list[str]:
    src = "\n".join(x for x in [pr.issue_title, pr.issue_body] if x) or "\n".join([pr.title, pr.description])
    src = re.sub(r"```.*?```", " ", src, flags=re.S)
    src = re.sub(r"<!--.*?-->", " ", src, flags=re.S)
    cands: list[str] = []
    for raw in src.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ">")) and len(line) < 4:
            continue
        bullet = re.match(r"^([-*+]|\d+[.)]|\[[ xX]\])\s+(.*)", line)
        if bullet:
            cands.append(bullet.group(2).strip())
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.strip("#* ").strip()
            if len(sent) > 12 and ACTION.search(sent):
                cands.append(sent)
    if not cands and (pr.issue_title or pr.title):
        cands = [pr.issue_title or pr.title]
    seen, out = set(), []
    for c in cands:
        c = re.sub(r"\s+", " ", c)[:220]
        k = c.lower()
        if k not in seen and len(_tokens(c)) >= 1:
            seen.add(k)
            out.append(c)
    # drop a requirement whose words are all contained in another one, e.g. the issue title
    # "Admin page needs user search" next to "The admin page should let admins search users by name."
    toks = [_tokens(c) for c in out]
    out = [c for i, c in enumerate(out) if not any(j != i and toks[i] < toks[j] for j in range(len(out)))]
    return out[:limit]


def is_test_requirement(text: str) -> bool:
    return bool(re.search(r"\btests?\b|\bunit test|coverage", text or "", re.I))


# words that appear on so many lines of code that sharing one says nothing about a requirement
CODE_WORDS = set(
    """return true false none null self this def class function const let var import export async await
try except catch finally raise throw print value values result data list dict string str int type error""".split()
)


def best_line(pr: PRInput, text: str, files: list[str] | None = None) -> tuple[str, int] | None:
    """The added line that shares the most words with `text` (ties: earliest), or None if none do.

    Generic code words ("return", "self", ...) are ignored, so "Return 400 when the query is empty"
    is not pinned to an unrelated `return` line.
    """
    want = _tokens(text) - CODE_WORDS
    if not want:
        return None
    cands = [pr.file(p) for p in files or []] or list(pr.files)
    best, best_n = None, 0
    for f in cands:
        if f is None or TEST_RE.search(f.path) or DOC_RE.search(f.path) or LOCK_RE.search(f.path):
            continue
        for no, t in f.added_lines():
            n = len(want & _tokens(t))
            if n > best_n:
                best, best_n = (f.path, no), n
    return best


def _offline(pr: PRInput) -> dict:
    reqs = extract_requirements(pr)
    file_tokens = {}
    for f in pr.files:
        text = f.path + "\n" + "\n".join(t for _, t in f.added_lines()) + "\n" + "\n".join(f.removed_lines())
        file_tokens[f.path] = _tokens(text)
    all_tokens = set().union(*file_tokens.values()) if file_tokens else set()
    has_tests = any(TEST_RE.search(f.path) for f in pr.files)
    items, used_files = [], set()
    for i, r in enumerate(reqs):
        rt = _tokens(r)
        if not rt:
            continue
        hit = rt & all_tokens
        cov = len(hit) / len(rt)
        if is_test_requirement(r) and not has_tests:
            cov = 0.0
        status = "covered" if cov >= 0.5 else ("partial" if cov >= 0.25 else "missing")
        files = sorted(file_tokens, key=lambda p: -len(rt & file_tokens[p]))
        files = [p for p in files if rt & file_tokens[p]][:3]
        used_files.update(files)
        items.append(
            {
                "id": f"R{i + 1}",
                "text": r,
                "status": status,
                "coverage": round(cov, 2),
                "files": files,
                "matched_terms": sorted(hit)[:8],
                "missing_terms": sorted(rt - all_tokens)[:8],
            }
        )
    unrequested = []
    if items and len(pr.files) >= 2:
        req_tokens = set().union(*(_tokens(it["text"]) for it in items))
        for f in pr.files:
            if TEST_RE.search(f.path) or DOC_RE.search(f.path) or LOCK_RE.search(f.path):
                continue
            ov = file_tokens[f.path] & req_tokens
            if not ov and f.path not in used_files:
                unrequested.append(
                    {
                        "file": f.path,
                        "reason": "no overlap with any requirement in the issue/description",
                        "lines": f.additions + f.deletions,
                    }
                )
    n = len(items)
    covered = sum(it["status"] == "covered" for it in items)
    return {
        "mode": "offline",
        "requirements": items,
        "unrequested": unrequested,
        "score": round(sum(it["coverage"] for it in items) / n, 2) if n else None,
        "summary": (f"{covered}/{n} requirements look covered" if n else "No explicit requirements found")
        + (f"; {len(unrequested)} file(s) look unrelated to the request" if unrequested else ""),
        "source": "issue" if pr.has_issue else "pr description",
    }


SYSTEM = """You are the intent checker of a code review agent. Compare what was REQUESTED (issue and PR
description) with what the DIFF actually does. Be concrete and literal; do not invent requirements.
Reply with JSON only:
{"requirements":[{"id":"R1","text":"<one requirement>","status":"covered|partial|missing",
  "files":["<path>"],"evidence":"<why, citing code>"}],
 "unrequested":[{"file":"<path>","reason":"<what was added that nobody asked for>"}],
 "summary":"<one sentence>"}"""


def check_intent(pr: PRInput, use_llm: bool) -> dict:
    base = _offline(pr)
    if not use_llm:
        return base
    diff = "\n\n".join(
        f"### {f.path} ({f.status}, +{f.additions}/-{f.deletions})\n{f.numbered_patch(120)}" for f in pr.files[:25]
    )[:14000]
    user = (
        f"ISSUE: {pr.issue_title}\n{pr.issue_body[:3000]}\n\nPR TITLE: {pr.title}\nPR DESCRIPTION:\n"
        f"{pr.description[:3000]}\n\nCANDIDATE REQUIREMENTS (from a parser, fix or extend):\n"
        + "\n".join(f"- {r['text']}" for r in base["requirements"])
        + f"\n\nDIFF:\n{diff}"
    )
    try:
        data, _ = llm.chat_json(SYSTEM, user, temperature=0.1, tag="intent")
        reqs = [r for r in (data or {}).get("requirements", []) if isinstance(r, dict) and r.get("text")]
        if not reqs:
            return base
        for i, r in enumerate(reqs):
            r["id"] = str(r.get("id") or f"R{i + 1}")
            r["status"] = (
                r.get("status", "partial") if r.get("status") in ("covered", "partial", "missing") else "partial"
            )
            files = r.get("files") or []
            files = [files] if isinstance(files, str) else files  # a model may return one path as a string
            r["files"] = list(dict.fromkeys(pr.file(p).path for p in files if isinstance(p, str) and pr.file(p)))
            r["coverage"] = {"covered": 1.0, "partial": 0.5, "missing": 0.0}[r["status"]]
        unreq = []
        for u in data.get("unrequested") or []:  # same tolerant path matching as the requirements
            f = pr.file(u.get("file", "")) if isinstance(u, dict) and isinstance(u.get("file"), str) else None
            if f is not None and f.path not in {x["file"] for x in unreq}:
                unreq.append({**u, "file": f.path})
        n = len(reqs)
        return {
            "mode": "llm",
            "requirements": reqs,
            "unrequested": unreq,
            "score": round(sum(r["coverage"] for r in reqs) / n, 2),
            "summary": data.get("summary") or base["summary"],
            "source": base["source"],
        }
    except Exception as e:
        base["llm_error"] = str(e)[:200]
        return base
    