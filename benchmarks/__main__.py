"""CLI: python -m benchmarks {download,stats,checkout} ...

python -m benchmarks download martian reviewbench   # fetch + convert to data/benchmarks/<name>/prs.jsonl
python -m benchmarks stats martian                   # counts + check against published numbers
python -m benchmarks checkout martian                # repos/martian/<pr>/ at each PR's head
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks import load_benchmark
from benchmarks.registry import get_benchmark


def _download(args) -> int:
    from benchmarks import martian, reviewbench

    mods = {"martian": martian, "reviewbench": reviewbench}
    for name in args.names:
        mods[name].download(refresh=args.refresh, limit=args.limit)
    return 0


def _stats(args) -> int:
    from benchmarks.stats import summarize, validate

    bad = 0
    for name in args.names:
        bench = get_benchmark(name)
        # Counting labels is not tuning; "eval" access lets us check the test set loaded correctly.
        examples = load_benchmark(name, purpose="eval")
        print(f"== {name} ({bench.split.upper()} set)")
        print(json.dumps(summarize(examples), indent=2))
        rows = validate(name, examples, bench.config.get("published", {}))
        print(f"-- check against published numbers ({sum(r[3] for r in rows)}/{len(rows)} match)")
        for key, exp, got, ok in rows:
            print(f"  {'ok ' if ok else 'BAD'} {key}: published={exp} loaded={got}")
        bad += sum(not r[3] for r in rows)
    return 1 if bad else 0


def _checkout(args) -> int:
    from benchmarks.checkout import CheckoutError, checkout

    examples = load_benchmark(args.name, purpose="eval" if get_benchmark(args.name).is_test else "tune")
    if args.id:
        examples = [e for e in examples if e.id in set(args.id)]
    failed = 0
    for i, ex in enumerate(examples):
        try:
            path = checkout(ex, ref=args.ref, force=args.force)
            print(f"[checkout] {i + 1}/{len(examples)} {ex.id} -> {path}", flush=True)
        except CheckoutError as e:
            failed += 1
            print(f"[checkout] FAILED {ex.id}: {e}", flush=True)
    print(f"[checkout] {len(examples) - failed}/{len(examples)} OK")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = ["martian", "reviewbench"]

    d = sub.add_parser("download", help="download and convert to the PRExample format")
    d.add_argument("names", nargs="+", choices=names)
    d.add_argument("--refresh", action="store_true", help="ignore cached API responses")
    d.add_argument("--limit", type=int, help="only the first N PRs (smoke test)")
    d.set_defaults(fn=_download)

    s = sub.add_parser("stats", help="print counts and compare with published numbers")
    s.add_argument("names", nargs="+", choices=names)
    s.set_defaults(fn=_stats)

    c = sub.add_parser("checkout", help="check out each PR's repo at head (or base)")
    c.add_argument("name", choices=names)
    c.add_argument("--id", action="append", help="only this PR id (repeatable)")
    c.add_argument("--ref", choices=["head", "base"], default="head")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=_checkout)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
