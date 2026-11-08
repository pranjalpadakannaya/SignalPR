"""Martian Code Review Bench (offline set) -> PRExample. DEV SET ONLY (issue #4).

Source: github.com/withmartian/code-review-benchmark, pinned in benchmarks.yaml.
- offline/golden_comments/<repo>.json: 50 PRs with 173 human-curated golden comments
  (comment text, severity Critical/High/Medium/Low, category). Golden comments have no file/line.
- offline/results/pr_labels.json: per-PR language and LLM-assigned labels (size, domain, risk, ...).

PR commits, diff, description and author come from the GitHub API. 31 PRs live in the upstream
repos (getsentry/sentry, grafana/grafana, ...); 19 were re-created in the ai-code-review-evaluation
org, and those keep a pointer to the upstream PR or commit in `original_url`.

    python -m benchmarks download martian
"""

from __future__ import annotations

import json
import re

from benchmarks import github
from benchmarks.common import cached, fetch_diff, finish, linked_issues, parse_pr_url
from benchmarks.registry import Benchmark, get_benchmark
from benchmarks.schema import KnownIssue, PRExample, norm_severity, write_jsonl

# Martian's scoring profiles: which golden categories count (offline/analysis/score_profiles.py)
PROFILES = {
    "strict": {"bug", "security", "concurrency", "data", "api"},
    "core": {"bug", "security", "concurrency", "data", "api", "perf", "test_gap", "doc_defect"},
    "all": {"bug", "security", "concurrency", "data", "api", "perf", "test_gap", "doc_defect", "style", "speculative"},
}
COMMIT_URL_RE = re.compile(r"github\.com/([^/]+/[^/]+)/commit/([0-9a-f]{7,40})")
REPO_LANGUAGE = {
    "sentry": "Python",
    "grafana": "Go",
    "cal_dot_com": "TypeScript",
    "discourse": "Ruby",
    "keycloak": "Java",
}


def fetch_sources(bench: Benchmark, refresh: bool = False) -> dict:
    """Download the pinned golden comments and PR labels into data/benchmarks/martian/raw/."""
    cfg = bench.config
    out = {}
    for repo in cfg["golden_repos"]:
        path = cfg["files"]["golden"].format(repo=repo)
        out[repo] = cached(
            bench.raw_dir / f"golden_{repo}.json",
            lambda p=path: {"items": json.loads(github.raw_file(cfg["source_repo"], cfg["source_commit"], p))},
            refresh,
        )["items"]
    labels = cached(
        bench.raw_dir / "pr_labels.json",
        lambda: json.loads(github.raw_file(cfg["source_repo"], cfg["source_commit"], cfg["files"]["labels"])),
        refresh,
    )
    return {"golden": out, "labels": labels}


def _commit(repo: str, sha: str) -> dict:
    try:
        d = github.rest(f"repos/{repo}/commits/{sha}")
    except github.GitHubError:
        return {}
    return {"message": d["commit"]["message"], "author": (d.get("author") or {}).get("login")}


def pr_id(url: str) -> str:
    repo, num = parse_pr_url(url)
    return f"martian/{repo.replace('/', '_')}_{num}"


def known_issues(ex_id: str, comments: list[dict]) -> list[KnownIssue]:
    out = []
    for i, c in enumerate(comments):
        cat = (c.get("category") or "").strip() or None
        out.append(
            KnownIssue(
                id=f"{ex_id}#{i}",
                body=c["comment"].strip(),
                severity=norm_severity(c.get("severity")),
                category=cat,
                label="tp",
                source="human_curated",
                extra={"profiles": [p for p, cats in PROFILES.items() if cat in cats]},
            )
        )
    return out


def convert_one(item: dict, golden_repo: str, labels: dict, bench: Benchmark, refresh: bool = False) -> PRExample:
    url = item["url"]
    repo, number = parse_pr_url(url)
    ex_id = pr_id(url)
    pr = cached(
        bench.raw_dir / "prs" / f"{ex_id.split('/')[1]}.json", lambda: github.pull_request(repo, number) or {}, refresh
    )
    if not pr.get("headRefOid"):
        raise RuntimeError(f"{url}: PR not found on GitHub")

    # Authorship and linked issues come from the original upstream PR when this is a re-created fork PR.
    original = item.get("original_url")
    upstream = parse_pr_url(original) if original else None
    source_pr = pr
    if upstream:
        source_pr = cached(
            bench.raw_dir / "prs" / f"upstream_{upstream[0].replace('/', '_')}_{upstream[1]}.json",
            lambda: github.pull_request(*upstream) or {},
            refresh,
        )
    elif repo.startswith("ai-code-review-evaluation/"):
        source_pr = None  # fork of an upstream commit; the fork author says nothing about who wrote it
    commit = COMMIT_URL_RE.search(original or "")
    upstream_commit = None
    if commit:
        upstream_commit = cached(
            bench.raw_dir / "prs" / f"upstream_commit_{commit.group(2)[:12]}.json",
            lambda: _commit(commit.group(1), commit.group(2)),
            refresh,
        )

    # Re-created PRs have placeholder bodies ("Test 1"); use the upstream PR body or commit message.
    description = pr.get("body") or ""
    if upstream and source_pr:
        description = source_pr.get("body") or description
    elif upstream_commit:
        description = upstream_commit.get("message") or description

    lab = labels.get(url, {})
    derived = lab.get("derived", {})
    ex = PRExample(
        id=ex_id,
        benchmark="martian",
        split=bench.split,
        repo=repo,
        pr_number=number,
        pr_url=url,
        clone_url=f"https://github.com/{repo}.git",
        base_sha=pr["baseRefOid"],
        head_sha=pr["headRefOid"],
        title=item.get("pr_title") or pr.get("title") or "",
        description=description,
        diff="",
        known_issues=known_issues(ex_id, item.get("comments", [])),
        metadata={
            "source_repo": golden_repo,
            "language": derived.get("language") or REPO_LANGUAGE.get(golden_repo),
            "original_url": original,
            "upstream_repo": upstream[0] if upstream else (None if source_pr is None else repo),
            "llm_pr_labels": lab.get("llm_pr_labels", {}),
            "golden_note": item.get("az_comment"),
        },
    )
    if upstream:
        ex.linked_issues = linked_issues(upstream[0], source_pr)
    elif source_pr:
        ex.linked_issues = linked_issues(repo, source_pr)
    diff_path = bench.raw_dir / "diffs" / f"{ex_id.split('/')[1]}.diff"
    if diff_path.exists() and not refresh:
        ex.diff = diff_path.read_text()
    else:
        ex.diff = fetch_diff(ex, [repo])
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(ex.diff)
    return finish(ex, source_pr or None)


def download(refresh: bool = False, limit: int | None = None) -> int:
    bench = get_benchmark("martian")
    src = fetch_sources(bench, refresh)
    items = [(repo, item) for repo, items in src["golden"].items() for item in items]
    items.sort(key=lambda x: pr_id(x[1]["url"]))
    examples, failed = [], []
    for i, (repo, item) in enumerate(items[:limit]):
        try:
            examples.append(convert_one(item, repo, src["labels"], bench, refresh))
            print(f"[martian] {i + 1}/{len(items)} {examples[-1].id}", flush=True)
        except Exception as e:  # keep going; report at the end
            failed.append((item["url"], str(e)))
            print(f"[martian] FAILED {item['url']}: {e}", flush=True)
    n = write_jsonl(examples, bench.prs_path)
    print(f"[martian] wrote {n} PRs -> {bench.prs_path}")
    for url, err in failed:
        print(f"  failed: {url}: {err}")
    return n
