"""Benchmarks in the common `PRExample` format (issues #4, #5).

One loader for every benchmark:

    from benchmarks import load_benchmark

    dev = load_benchmark("martian", purpose="tune")        # DEV set: tune on this
    test = load_benchmark("reviewbench", purpose="eval")   # TEST set: final evaluation only
    load_benchmark("reviewbench", purpose="tune")          # -> TestSetAccessError

Download first with `python -m benchmarks download martian reviewbench` (see benchmarks/README.md).
"""

from __future__ import annotations

from benchmarks.registry import TestSetAccessError, all_benchmarks, check_access, get_benchmark
from benchmarks.schema import ChangedFile, KnownIssue, LinkedIssue, PRExample, read_jsonl


def load_benchmark(name: str, purpose: str, subset: str | None = None) -> list[PRExample]:
    """Load every PR of a benchmark.

    purpose: "tune" (prompt/threshold tuning, ablations, debugging) or "eval" (final scoring).
             Test-only benchmarks refuse "tune".
    subset:  reviewbench only: "test_subset" for ReviewBench's own 25-PR smoke-test slice.
    """
    bench = check_access(name, purpose)
    if not bench.prs_path.exists():
        raise FileNotFoundError(f"{bench.prs_path} not found. Run `python -m benchmarks download {name}` first.")
    examples = read_jsonl(bench.prs_path)
    if subset == "test_subset":
        examples = [e for e in examples if e.metadata.get("in_test_subset")]
    elif subset is not None:
        raise ValueError(f"unknown subset {subset!r}")
    return examples


__all__ = [
    "ChangedFile",
    "KnownIssue",
    "LinkedIssue",
    "PRExample",
    "TestSetAccessError",
    "all_benchmarks",
    "check_access",
    "get_benchmark",
    "load_benchmark",
]
