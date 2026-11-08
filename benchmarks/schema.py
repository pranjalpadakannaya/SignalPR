"""The common `PRExample` format every benchmark is converted into (issue #4).

Every system (SignalPR agent, baselines) and the evaluation harness read PRs through this one
schema, so they never have to know which benchmark a PR came from.

Stored on disk as JSON Lines, one PRExample per line (`data/benchmarks/<name>/prs.jsonl`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

# Severities are lower-cased across benchmarks. Martian uses Critical/High/Medium/Low,
# ReviewBench uses high/medium/low. Weights are used by severity-weighted recall (#7).
SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_WEIGHT = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}

# PR size buckets (added + removed lines), same cut points ReviewBench reports.
SIZE_BUCKETS = ((50, "xs"), (200, "s"), (500, "m"), (1000, "l"))


def size_bucket(lines_changed: int) -> str:
    for limit, name in SIZE_BUCKETS:
        if lines_changed <= limit:
            return name
    return "xl"


def norm_severity(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    return v if v in SEVERITIES else None


@dataclass
class ChangedFile:
    path: str
    status: str = "modified"  # added | modified | removed | renamed
    old_path: str | None = None
    additions: int = 0
    deletions: int = 0
    binary: bool = False


@dataclass
class LinkedIssue:
    number: int | None
    title: str
    body: str
    url: str | None = None


@dataclass
class KnownIssue:
    """One ground-truth finding for a PR.

    `label` is "tp" for real issues. ReviewBench also ships judged false positives ("fp"): a system
    comment that matches one of those counts against its grounded precision, so we keep them.
    `file`/`start_line`/`end_line` are on the head commit and may be None (Martian has no locations).
    """

    id: str
    body: str
    severity: str | None = None
    category: str | None = None
    file: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    label: str = "tp"
    source: str | None = None  # who found it: human reviewer, llm_review, tool, ...
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PRExample:
    id: str  # "<benchmark>/<owner>_<repo>_<pr>", stable across re-downloads
    benchmark: str  # "martian" | "reviewbench"
    split: str  # "dev" | "test" (see benchmarks/registry.py)
    repo: str  # "owner/name" the PR was opened against
    pr_number: int | None
    pr_url: str
    clone_url: str  # a git remote that holds base_sha and head_sha
    base_sha: str
    head_sha: str
    title: str
    description: str
    diff: str  # unified diff base...head
    files: list[ChangedFile] = field(default_factory=list)
    linked_issues: list[LinkedIssue] = field(default_factory=list)
    known_issues: list[KnownIssue] = field(default_factory=list)
    # language, lines_added, lines_removed, size_bucket, author, author_kind (agent|human|unknown),
    # agent, plus benchmark-specific labels
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def real_issues(self) -> list[KnownIssue]:
        return [k for k in self.known_issues if k.label == "tp"]

    @property
    def linked_issue_text(self) -> str:
        return "\n\n".join(f"#{i.number} {i.title}\n{i.body}".strip() for i in self.linked_issues)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PRExample:
        d = dict(d)
        d["files"] = [ChangedFile(**_known(ChangedFile, f)) for f in d.get("files", [])]
        d["linked_issues"] = [LinkedIssue(**_known(LinkedIssue, i)) for i in d.get("linked_issues", [])]
        d["known_issues"] = [KnownIssue(**_known(KnownIssue, k)) for k in d.get("known_issues", [])]
        return cls(**_known(cls, d))


def _known(cls, d: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def write_jsonl(examples: Iterable[PRExample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path) -> list[PRExample]:
    with path.open(encoding="utf-8") as f:
        return [PRExample.from_dict(json.loads(line)) for line in f if line.strip()]
