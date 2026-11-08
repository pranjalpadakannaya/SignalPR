"""python -m tracking log results/eval/<name>.json [--prompt-version v1] [--notes "..."]
python -m tracking list

Records an evaluation run in results/runs/<run_id>/ and adds it to results/runs/runs.tsv.
"""

from __future__ import annotations

import argparse
import sys

from tracking.log import SHEET, list_runs, log_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tracking", description="Log and list experiment runs (issue #3)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    lg = sub.add_parser("log", help="record a result file from python -m eval")
    lg.add_argument("result", help="e.g. results/eval/martian.json")
    lg.add_argument("--prompt-version", default="inline", help='prompt files used, e.g. "v1"')
    lg.add_argument("--seed", type=int, default=0)
    lg.add_argument("--owner", help="default: your git user.name")
    lg.add_argument("--notes", default="", help="what this run was for")

    sub.add_parser("list", help="show the results sheet")

    a = ap.parse_args(argv)
    if a.cmd == "log":
        run_dir = log_run(a.result, prompt_version=a.prompt_version, seed=a.seed, owner=a.owner, notes=a.notes)
        print(f"logged run: {run_dir}\nsheet:      {SHEET}")
        return 0

    rows = list_runs()
    if not rows:
        print("no runs logged yet")
        return 0
    for r in rows:
        print(f"{r['run_id']:34s} {r['system'][:32]:32s} P={r['precision']:7s} R={r['recall']:7s} F1={r['f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
