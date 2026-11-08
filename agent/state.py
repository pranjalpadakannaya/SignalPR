"""Data passed between the review steps: the PR under review, tool evidence, and candidate comments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tools.diff import FileDiff, parse_unified_diff

SEVERITIES = ("critical", "medium", "low")
SEVERITY_WEIGHT = {"critical": 1.0, "medium": 0.6, "low": 0.25}
SEVERITY_ORDER = {"critical": 3, "medium": 2, "low": 1}
CATEGORIES = ("correctness", "security", "design", "performance", "intent", "tests")


@dataclass
class PRInput:
    files: list[FileDiff]
    title: str = ""
    description: str = ""
    issue_title: str = ""
    issue_body: str = ""
    agent: str = "unknown"
    repo: str = ""
    repo_language: str | None = None
    stars: int = 0
    forks: int = 0
    url: str = ""
    number: int | None = None
    head_sha: str = ""
    repo_path: str | None = None  # local checkout: enables test runs and repo-wide usage lookup
    test_cmd: str | None = None

    @classmethod
    def from_diff(cls, diff: str, **meta) -> "PRInput":
        return cls(files=parse_unified_diff(diff), **meta)

    @property
    def has_issue(self) -> bool:
        return bool(self.issue_title or self.issue_body)

    @property
    def issue_text(self) -> str:
        return "\n".join(x for x in (self.issue_title, self.issue_body) if x)

    def file(self, path: str) -> FileDiff | None:
        path = (path or "").strip()
        for f in self.files:
            if f.path == path:
                return f
        clean = path.lstrip("./").removeprefix("a/").removeprefix("b/")
        for f in self.files:  # tolerate "b/..." prefixes or partial paths written by an LLM
            if clean and (f.path == clean or f.path.endswith("/" + clean) or clean.endswith("/" + f.path)):
                return f
        return None

    def meta(self) -> dict:
        """PR-level fields the risk model uses."""
        return {
            "agent": self.agent,
            "title": self.title,
            "body": self.description,
            "issue_text": self.issue_text,
            "stars": self.stars,
            "forks": self.forks,
            "repo_language": self.repo_language,
        }


@dataclass
class Evidence:
    id: str
    file: str
    line: int | None
    tool: str  # ruff | eslint | semgrep | lizard | complexity | rules | tests | duplication | usage
    rule: str
    category: str
    severity: str
    message: str
    snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Comment:
    file: str
    line: int | None
    category: str
    severity: str
    title: str
    body: str
    source: str  # "<specialist>/llm" or "<specialist>/tools" or "baseline/..."
    evidence: list[str] = field(default_factory=list)
    quote: str = ""
    run: int = 0
    suggestion: str = ""
    # filled in by the verifier and the ranker
    support: float = 0.0  # share of runs that produced this comment
    verified_by: list[str] = field(default_factory=list)
    drop_reason: str = ""
    risk: float = 0.0
    score: float = 0.0
    score_parts: dict = field(default_factory=dict)
    snippet: list = field(default_factory=list)

    @property
    def from_llm(self) -> bool:
        return self.source.endswith("/llm")

    def to_dict(self) -> dict:
        return asdict(self)
