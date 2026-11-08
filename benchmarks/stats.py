"""Counts per benchmark and a check against the benchmark's published numbers.

python -m benchmarks stats martian
python -m benchmarks stats reviewbench      # exits non-zero if any count disagrees
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmarks.schema import PRExample, size_bucket


def _count(values) -> dict[str, int]:
    return dict(Counter(v for v in values if v is not None).most_common())


def summarize(examples: list[PRExample]) -> dict[str, Any]:
    issues = [k for e in examples for k in e.known_issues]
    tps = [k for k in issues if k.label == "tp"]
    return {
        "prs": len(examples),
        "repos": len({e.repo.lower() for e in examples}),
        "findings": len(issues),
        "true_positives": len(tps),
        "by_severity_tp": _count(k.severity for k in tps),
        "by_category_tp": _count(k.category for k in tps),
        "by_source_type_tp": _count(k.source for k in tps),
        "by_language": _count(e.metadata.get("language") for e in examples),
        "by_size_bucket": _count(e.metadata.get("size_bucket") for e in examples),
        "by_author_kind": _count(e.metadata.get("author_kind") for e in examples),
        "by_agent": _count(e.metadata.get("agent") for e in examples),
        "with_linked_issue": sum(bool(e.linked_issues) for e in examples),
        "empty_diff": [e.id for e in examples if not e.diff.strip()],
    }


def _published_view(name: str, examples: list[PRExample]) -> dict[str, Any]:
    """The same quantities the benchmark authors published, computed from our loaded data."""
    s = summarize(examples)
    view: dict[str, Any] = {k: s[k] for k in ("prs", "repos", "findings", "true_positives", "by_language")}
    if name == "martian":
        view["golden_comments"] = s["findings"]
        view["repos"] = len({e.metadata.get("source_repo", "").split("-")[0] for e in examples})
        view["by_profile"] = {
            p: sum(p in k.extra.get("profiles", []) for e in examples for k in e.known_issues)
            for p in ("strict", "core", "all")
        }
    if name == "reviewbench":
        # ReviewBench buckets PR size by the manifest's line counts, not by our parsed diff
        view["by_size_bucket"] = _count(
            size_bucket((e.metadata.get("manifest_lines_added") or 0) + (e.metadata.get("manifest_lines_removed") or 0))
            for e in examples
        )
        view["by_source_type_tp"] = s["by_source_type_tp"]
        subset = [e for e in examples if e.metadata.get("in_test_subset")]
        ts = summarize(subset)
        view["test_subset"] = {k: ts[k] for k in ("prs", "true_positives", "by_severity_tp", "by_category_tp")}
    return view


def compare(published: dict[str, Any], actual: dict[str, Any], prefix: str = "") -> list[tuple[str, Any, Any, bool]]:
    """Flatten published vs actual into (key, expected, actual, ok) rows."""
    rows = []
    for key, exp in published.items():
        got = actual.get(key) if isinstance(actual, dict) else None
        name = f"{prefix}{key}"
        if isinstance(exp, dict):
            rows += compare(exp, got or {}, prefix=f"{name}.")
        else:
            rows.append((name, exp, got, exp == got))
    return rows


def validate(name: str, examples: list[PRExample], published: dict[str, Any]) -> list[tuple[str, Any, Any, bool]]:
    return compare(published, _published_view(name, examples))
