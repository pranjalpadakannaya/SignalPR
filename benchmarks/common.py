"""Steps shared by the benchmark converters: diff, changed files, linked issues, PR metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from benchmarks import github
from benchmarks.authorship import classify
from benchmarks.schema import ChangedFile, LinkedIssue, PRExample, size_bucket
from tools.diff import parse_unified_diff

PR_URL_RE = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
CLOSING_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|addresse[sd]|address(?:ing)?|related to|refs?)\b:?\s+"
    r"(?:(?:https?://github\.com/)?([\w.-]+/[\w.-]+)(?:/issues/|#)|#)(\d+)",
    re.I,
)


def parse_pr_url(url: str) -> tuple[str, int] | None:
    m = PR_URL_RE.search(url or "")
    return (m.group(1), int(m.group(2))) if m else None


def files_from_diff(diff: str) -> list[ChangedFile]:
    return [
        ChangedFile(
            path=f.path,
            status=f.status,
            old_path=f.old_path if f.old_path and f.old_path != f.path else None,
            additions=f.additions,
            deletions=f.deletions,
            binary=f.binary,
        )
        for f in parse_unified_diff(diff)
    ]


def fetch_diff(ex: PRExample, repos: list[str]) -> str:
    """base...head diff: GitHub compare API on each candidate repo, then a local git fallback."""
    for repo in dict.fromkeys(repos):
        diff = github.compare_diff(repo, ex.base_sha, ex.head_sha)
        if diff:
            return diff
    from benchmarks.checkout import git_diff

    return git_diff(ex)


def linked_issues(repo: str, pr: dict[str, Any] | None) -> list[LinkedIssue]:
    """Issues the PR closes (GitHub's own link), else 'fixes #N' style references in the body."""
    if not pr:
        return []
    found = {i["number"]: i for i in pr.get("linked_issues") or []}
    if not found:
        for m in CLOSING_RE.finditer(pr.get("body") or ""):
            ref_repo, num = (m.group(1) or repo), int(m.group(2))
            if ref_repo.lower() == repo.lower() and num not in found:
                got = github.issue(repo, num)
                if got:
                    found[num] = got
    return [
        LinkedIssue(number=i["number"], title=i.get("title") or "", body=i.get("body") or "", url=i.get("url"))
        for i in found.values()
    ]


def finish(ex: PRExample, pr: dict[str, Any] | None) -> PRExample:
    """Fill files, size and authorship metadata once diff and description are set."""
    ex.files = files_from_diff(ex.diff)
    added = sum(f.additions for f in ex.files)
    removed = sum(f.deletions for f in ex.files)
    ex.metadata.update(
        lines_added=added,
        lines_removed=removed,
        files_changed=len(ex.files),
        size_bucket=size_bucket(added + removed),
        has_linked_issue=bool(ex.linked_issues),
        **classify(pr),
    )
    return ex


def cached(path: Path, build: Callable[[], dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
    """Per-PR JSON cache so an interrupted download resumes where it stopped."""
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))
    return data
