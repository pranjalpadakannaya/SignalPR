"""Step 4 of the review: verifier. A candidate comment survives only if it
  (a) points to a specific changed line of a file in the diff,
  (b) is backed by evidence: a tool finding at that location, or code it quotes that really
      is on that line (and any `identifiers` it mentions actually appear in the change), and
  (c) for LLM comments, shows up in most of the repeated runs (self-consistency).
Near-duplicates are merged. Every dropped comment keeps its reason, so the UI can show
what was filtered out and why.
"""

from __future__ import annotations

import math
import re

from agent.state import SEVERITY_ORDER, Comment, Evidence, PRInput

_norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().strip("+").strip())  # noqa: E731
BACKTICK = re.compile(r"`([A-Za-z_][\w.]{2,60})`")


def _locate(f, quote: str) -> int | None:
    q = _norm(quote)
    if len(q) < 4:
        return None
    for no, t in f.added_lines():
        if q in _norm(t) or (len(_norm(t)) > 8 and _norm(t) in q):
            return no
    return None


def _anchor(c: Comment, pr: PRInput) -> str | None:
    f = pr.file(c.file)
    if f is None:
        return "file is not part of this diff"
    c.file = f.path
    changed = f.changed_lines()
    if not changed:
        return "file has no added lines to comment on"
    if c.line is None or c.line not in changed:
        loc = _locate(f, c.quote) if c.quote else None
        if loc is not None:
            c.line = loc
        elif c.line is not None:
            near = min(changed, key=lambda n: abs(n - c.line))
            if abs(near - c.line) <= 3:
                c.line = near
            else:
                return f"line {c.line} is not a changed line"
        else:
            return "no line given and quoted code not found"
    c.snippet = f.snippet(c.line, 2)
    return None


def _support(c: Comment, pr: PRInput, ev: dict[str, Evidence]) -> str | None:
    f = pr.file(c.file)
    is_llm = c.from_llm
    tool_ok = []
    for eid in c.evidence:
        if eid.startswith("intent:"):
            tool_ok.append(eid)
            continue
        e = ev.get(eid)
        if e and pr.file(e.file) is f and (e.line is None or abs((e.line or 0) - c.line) <= 3):
            tool_ok.append(eid)
    if not tool_ok:  # a tool finding on the same line, same category and similar wording also counts
        for e in ev.values():
            if (
                pr.file(e.file) is f
                and e.line == c.line
                and e.category == c.category
                and _similar_text(e.message, c.title + " " + c.body)
            ):
                tool_ok.append(e.id)
                break
    c.evidence = tool_ok or c.evidence
    if tool_ok:
        c.verified_by.append("tool evidence")
    q = _norm(c.quote)
    if q:
        near = " ".join(_norm(f.new_line_text(n) or "") for n in range(c.line - 1, c.line + 2))
        if q in near or (len(q) > 12 and _norm(f.new_line_text(c.line) or "") in q):
            c.verified_by.append("quote matches code")
        elif not tool_ok or is_llm:
            return "quoted code is not on that line (likely hallucinated)"
    idents = BACKTICK.findall(c.body + " " + c.title)
    if idents and is_llm:
        hay = (f.content or "") + "\n" + f.patch_text()
        found = [i for i in idents if i.split(".")[-1].split("(")[0] in hay]
        if not found:
            return f"mentions `{idents[0]}`, which does not appear in the change"
    if not c.verified_by:
        return "no supporting evidence (no tool finding, no matching quote)"
    return None


def _words(c: Comment) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", (c.title + " " + c.body[:200]).lower())} - {
        "this",
        "that",
        "with",
        "from",
        "code",
        "line",
        "file",
        "found",
        "rules",
        "security",
        "correctness",
        "design",
    }


def _similar(a: Comment, b: Comment) -> bool:
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.3


def _similar_text(x: str, y: str) -> bool:
    wa = set(re.findall(r"[a-z]{4,}", x.lower()))
    wb = set(re.findall(r"[a-z]{4,}", y.lower()))
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= 0.34


def _same(a: Comment, b: Comment) -> bool:
    return a.file == b.file and a.line is not None and b.line is not None and abs(a.line - b.line) <= 3


def verify(
    pr: PRInput, candidates: list[Comment], evidence: list[Evidence], runs: int, use_consistency: bool = True
) -> dict:
    ev = {e.id: e for e in evidence}
    kept: list[Comment] = []
    dropped: list[Comment] = []
    for c in candidates:
        reason = _anchor(c, pr) or _support(c, pr, ev)
        if reason:
            c.drop_reason = reason
            dropped.append(c)
        else:
            kept.append(c)

    # ---- self-consistency over repeated LLM runs (per specialist) ----
    llm_kept = [c for c in kept if c.from_llm]
    tool_kept = [c for c in kept if not c.from_llm]
    for c in tool_kept:
        c.support = 1.0
    need = math.ceil(runs / 2 + 1e-9) if runs > 1 else 1  # majority of runs
    clusters: list[list[Comment]] = []
    for c in llm_kept:
        for cl in clusters:
            if (
                cl[0].source == c.source
                and _same(cl[0], c)
                and (_similar(cl[0], c) or (_norm(cl[0].quote) == _norm(c.quote) and cl[0].severity == c.severity))
            ):
                cl.append(c)
                break
        else:
            clusters.append([c])
    consistent = []
    for cl in clusters:
        n_runs = len({c.run for c in cl})
        best = max(cl, key=lambda c: (SEVERITY_ORDER[c.severity], len(c.body)))
        best.support = n_runs / max(runs, 1)
        if use_consistency and n_runs < need and "tool evidence" not in best.verified_by:
            best.drop_reason = f"appeared in only {n_runs}/{runs} runs"
            dropped.append(best)
            continue
        if n_runs >= need:
            best.verified_by.append(f"{n_runs}/{runs} runs agree")
        consistent.append(best)

    # ---- merge LLM comments with tool comments at the same spot ----
    final: list[Comment] = []
    for c in consistent + tool_kept:
        both_tools = not c.from_llm
        dup = next(
            (
                k
                for k in final
                if k.file == c.file
                and (
                    (k.line == c.line and both_tools and not k.from_llm and k.category == c.category)
                    or (_same(k, c) and (set(k.evidence) & set(c.evidence) or _similar(k, c)))
                )
            ),
            None,
        )
        if dup is None:
            final.append(c)
            continue
        # keep the richer text, union the evidence
        pref = lambda x: (x.from_llm, bool(x.suggestion), len(x.body))  # noqa: E731
        a, b = (dup, c) if pref(dup) >= pref(c) else (c, dup)
        a.evidence = sorted(set(a.evidence) | set(b.evidence))
        a.verified_by = sorted(set(a.verified_by) | set(b.verified_by))
        a.severity = max([a.severity, b.severity], key=SEVERITY_ORDER.get)
        a.support = max(a.support, b.support)
        a.suggestion = a.suggestion or b.suggestion
        extra = b.body.split(" Found by")[0].strip()
        if a.category != b.category and extra and extra[:60].lower() not in a.body.lower() and not _similar(a, b):
            a.body += f"\n\nAlso flagged ({b.category}): {extra}"
        b.drop_reason = "merged into a comment on the same line"
        dropped.append(b)
        if a is not dup:
            final[final.index(dup)] = a
    return {"kept": final, "dropped": dropped}
