"""Scoring a reviewer's comments against a PR's known issues.

Known issues are dicts {id, body, severity, category, file, start_line, end_line, label}, the fields of
benchmarks.schema.KnownIssue. `label` is "tp" for real issues; ReviewBench also ships judged false
positives ("fp").

Matching (greedy, in the order comments are shown, each known issue matched at most once): a comment
matches an issue in the same file whose line range (widened by `line_tol`) contains the comment's line,
when its wording or category fits. Issues with no location (Martian) are matched on wording alone, or
by the LLM judge (eval/judge.py).

Metrics (pooled over PRs, 95% bootstrap CIs resampling PRs):
  precision   grounded: comments matching a real issue / all comments (matching a judged FP or nothing
              counts against it)
  recall, F1  real issues found / all real issues
  severity-weighted recall   weights from benchmarks.schema.SEVERITY_WEIGHT (critical 4 ... low 1)
  recall@k    real issues found within the first k comments (tests the ranking), k = 1, 3, 5
  comments per real issue    how many comments a person reads to find one real problem
"""

from __future__ import annotations

import re

import numpy as np

from benchmarks.schema import SEVERITY_WEIGHT

KS = (1, 3, 5)
# benchmark categories (ReviewBench, Martian, demo set) -> reviewer categories that may raise them
TYPE_MAP = {
    "security": {"security"},
    "correctness": {"correctness", "performance"},
    "reliability": {"correctness"},
    "bug": {"correctness"},
    "concurrency": {"correctness", "performance"},
    "data": {"correctness"},
    "performance": {"performance", "correctness"},
    "perf": {"performance", "correctness"},
    "api": {"design", "correctness"},
    "api-architecture": {"design"},
    "maintainability": {"design", "intent"},
    "design": {"design", "intent"},
    "style": {"design"},
    "documentation": {"design"},
    "doc_defect": {"design"},
    "testing": {"tests", "design"},
    "test_gap": {"tests", "design"},
    "tests": {"tests"},
}
_STOP = set("the a an and or of to in on for with is are be this that it as by from not no at".split())


def _tok(s: str) -> set:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", s.lower().replace("_", " ")) if w not in _STOP}


def _same_file(a: str, b: str) -> bool:
    a, b = (a or "").lstrip("./"), (b or "").lstrip("./")
    return bool(a and b) and (a == b or a.endswith("/" + b) or b.endswith("/" + a))


def _in_range(issue: dict, line, tol: int) -> bool:
    start = issue.get("start_line")
    if not start or not line:
        return True
    end = issue.get("end_line") or start
    return int(start) - tol <= int(line) <= int(end) + tol


def match(
    comments: list[dict], issues: list[dict], line_tol: int = 5, min_sim: float = 0.12, text_min: float = 0.3
) -> list[int | None]:
    """For each comment (in shown order), the index of the known issue it matches, or None."""
    used, out = set(), []
    for c in comments:
        best, best_s = None, 0.0
        ctext = _tok(" ".join(str(c.get(k) or "") for k in ("title", "body", "quote")))
        for gi, g in enumerate(issues):
            if gi in used:
                continue
            located = bool(g.get("file"))
            if located and not (_same_file(g["file"], c.get("file", "")) and _in_range(g, c.get("line"), line_tol)):
                continue
            gt = _tok(g.get("body", ""))
            sim = len(gt & ctext) / max(1, len(gt))
            type_ok = c.get("category") in TYPE_MAP.get(g.get("category") or "", {g.get("category")})
            ok = (sim >= min_sim or type_ok) if located else sim >= text_min
            s = sim + (0.25 if type_ok else 0)
            if ok and s > best_s:
                best, best_s = gi, s
        if best is not None:
            used.add(best)
        out.append(best)
    return out


def pr_metrics(comments: list[dict], issues: list[dict], matches: list[int | None] | None = None) -> dict:
    """Per-PR counts. `matches` (e.g. from the LLM judge) replaces the lexical matcher when given."""
    m = match(comments, issues) if matches is None else matches
    real = [i for i, g in enumerate(issues) if g.get("label", "tp") == "tp"]
    hit = [x for x in m if x is not None and issues[x].get("label", "tp") == "tp"]
    w = lambda g: SEVERITY_WEIGHT.get(g.get("severity") or "", 1.0)  # noqa: E731
    out = {
        "n_comments": len(comments),
        "n_gold": len(real),
        "tp": len(set(hit)),
        "matched_fp": sum(1 for x in m if x is not None and issues[x].get("label", "tp") != "tp"),
        "sev_found": sum(w(issues[i]) for i in set(hit)),
        "sev_total": sum(w(issues[i]) for i in real),
    }
    for k in KS:
        out[f"found@{k}"] = len({x for x in m[:k] if x is not None and issues[x].get("label", "tp") == "tp"})
    return out


def _pool(rows: list[dict]) -> dict:
    c = sum(r["n_comments"] for r in rows)
    g = sum(r["n_gold"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    p, rc = (tp / c if c else 0.0), (tp / g if g else 0.0)
    out = {
        "precision": p,
        "recall": rc,
        "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "sev_weighted_recall": sum(r["sev_found"] for r in rows) / max(1e-9, sum(r["sev_total"] for r in rows)),
        "comments_per_real_issue": c / tp if tp else None,
        "comments": c,
        "true_positives": tp,
        "matched_judged_fp": sum(r.get("matched_fp", 0) for r in rows),
        "known_issues": g,
    }
    for k in KS:
        out[f"recall@{k}"] = sum(r[f"found@{k}"] for r in rows) / g if g else 0.0
    return out


def aggregate(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Pooled metrics plus 95% bootstrap CIs (resampling PRs) for precision, recall and F1."""
    point = _pool(rows)
    if not rows:
        return point | {"ci95": {}}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in ("precision", "recall", "f1")}
    for _ in range(n_boot):
        a = _pool([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        for k in boots:
            boots[k].append(a[k])
    point["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    return point
