"""python -m eval --bench demo|martian|reviewbench|<prs.jsonl> [options]

    python -m eval --bench demo                                   # harness check, no model needed
    python -m eval --bench martian --repeats 3                    # DEV set (judge: LLM, issues have no lines)
    python -m eval --bench reviewbench --subset test_subset       # TEST set, frozen config only

Writes results/eval/<name>.json and results/eval/<name>.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.llm_client import ROOT
from eval.run import SYSTEMS, run_benchmark, to_markdown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval", description="Run SignalPR, baselines and ablations")
    ap.add_argument("--bench", default="demo")
    ap.add_argument("--subset", help="reviewbench: test_subset (ReviewBench's 25-PR slice)")
    ap.add_argument(
        "--profile", choices=["strict", "core", "all"], help="martian scoring profile (Martian's default: core)"
    )
    ap.add_argument("--systems", nargs="*", choices=list(SYSTEMS), help="default: all")
    ap.add_argument("--repeats", type=int, default=1, help="repeat every system")
    ap.add_argument("--runs", type=int, default=3, help="self-consistency samples per specialist")
    ap.add_argument("--limit", type=int, help="only the first N PRs (smoke test)")
    ap.add_argument(
        "--judge",
        choices=["auto", "lexical", "llm"],
        default="auto",
        help="how comments are matched to known issues (auto: LLM when issues have no location)",
    )
    ap.add_argument("--out", help="output JSON (default results/eval/<name>.json)")
    a = ap.parse_args(argv)
    name = a.bench if a.bench in ("demo", "martian", "reviewbench") else Path(a.bench).stem
    name += f"_{a.subset}" if a.subset else ""
    out = Path(a.out) if a.out else ROOT / "results" / "eval" / f"{name}.json"
    res = run_benchmark(
        a.bench,
        a.systems,
        a.repeats,
        a.runs,
        out,
        a.limit,
        verbose=True,
        judge=a.judge,
        subset=a.subset,
        profile=a.profile,
    )
    out.with_suffix(".md").write_text(to_markdown(res))
    print(f"\nwrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
