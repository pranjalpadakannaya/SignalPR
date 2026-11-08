"""Command line for the reviewer.

    python -m agent samples                                   # list demo PRs
    python -m agent review --sample flask-user-search         # review a demo PR
    python -m agent review --diff change.diff --title "Add search" --issue issue.md
    python -m agent review --url https://github.com/owner/repo/pull/123 [--post]
    python -m agent review ... --mode single_prompt|tools_only|size_rank --runs 3 --json out.json
    python -m agent health                                    # is the LLM server reachable?

LLM settings come from the environment (see agent/llm_client.py), e.g.
    SIGNALPR_LLM_PRESET=qwen-7b SIGNALPR_LLM_BACKEND=ollama python -m agent review --sample cart-rounding
Without a reachable server the review runs on tool evidence only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent import llm
from agent.pipeline import MODES, review
from agent.samples import list_samples, load_sample, pr_from_dict
from agent.state import PRInput


def _print(res: dict) -> None:
    p = res["pr"]
    print(f"\n{p['title'] or '(untitled)'}  ·  {p['files']} files, +{p['additions']}/-{p['deletions']}")
    print(f"LLM: {res['llm']['model'] if res['llm']['used'] else 'off'} ({res['llm']['reason'] or 'ok'})")
    if res["intent"].get("summary"):
        print(f"Intent: {res['intent']['summary']}")
    for r in res["intent"].get("requirements", [])[:8]:
        print(f"  [{r['status']:>7}] {r['text'][:100]}")
    print(
        f"\n{res['stats']['shown']} comment(s), {res['stats']['dropped']} filtered, "
        f"{res['stats']['evidence']} evidence item(s), {res['stats']['seconds']} s"
    )
    for i, c in enumerate(res["comments"], 1):
        print(f"\n{i}. [{c['severity']}] {c['file']}:{c['line']}  {c['title']}")
        print("   " + c["body"].replace("\n", "\n   ")[:600])
        if c["suggestion"]:
            print(f"   -> {c['suggestion'][:300]}")
        print(f"   score {c['score']:.2f}  risk {c['risk']:.1%}  evidence: {', '.join(c['verified_by'])}")
    if res["file_risk"]:
        print("\nFile risk:")
        for r in res["file_risk"][:8]:
            print(f"  {r['risk']:6.1%}  {r['file']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent", description="SignalPR code reviewer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("samples", help="list demo PRs")
    sub.add_parser("health", help="check the LLM server")
    rv = sub.add_parser("review", help="review a PR")
    src = rv.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample", help="demo PR id (see `samples`)")
    src.add_argument("--diff", help="unified diff file ('-' for stdin)")
    src.add_argument("--url", help="GitHub pull request URL")
    rv.add_argument("--title", default="")
    rv.add_argument("--description", default="")
    rv.add_argument("--issue", help="file with the issue text (first line = title)")
    rv.add_argument("--agent", default="unknown")
    rv.add_argument("--repo-path", help="local checkout (enables test runs and repo-wide usage lookup)")
    rv.add_argument("--test-cmd", help="command that runs the project's tests, e.g. 'pytest -q'")
    rv.add_argument("--mode", default="signalpr", choices=MODES)
    rv.add_argument("--runs", type=int, default=3)
    rv.add_argument("--no-llm", action="store_true")
    for step in ("intent", "evidence", "verifier", "consistency", "ranker"):
        rv.add_argument(f"--no-{step}", action="store_true", help=f"ablation: switch off the {step} step")
    rv.add_argument("--max-comments", type=int, default=8)
    rv.add_argument("--json", help="write the full result to this file")
    rv.add_argument("--post", action="store_true", help="post the review on the GitHub PR (needs GITHUB_TOKEN)")
    a = ap.parse_args(argv)

    if a.cmd == "samples":
        for s in list_samples():
            print(f"{s['id']:<20} {s['title']}  ({s['known_issues']} known issues)")
        return 0
    if a.cmd == "health":
        print(json.dumps(llm.status(force=True), indent=2))
        return 0

    if a.sample:
        pr = pr_from_dict(load_sample(a.sample))
    elif a.url:
        from agent.github import fetch_pr

        pr = fetch_pr(a.url)
    else:
        diff = sys.stdin.read() if a.diff == "-" else Path(a.diff).read_text()
        issue_title = issue_body = ""
        if a.issue:
            lines = Path(a.issue).read_text().splitlines()
            issue_title, issue_body = (lines[0] if lines else ""), "\n".join(lines[1:])
        pr = PRInput.from_diff(
            diff,
            title=a.title,
            description=a.description,
            issue_title=issue_title,
            issue_body=issue_body,
            agent=a.agent,
        )
    pr.repo_path = a.repo_path or pr.repo_path
    pr.test_cmd = a.test_cmd or pr.test_cmd
    opts = {"mode": a.mode, "runs": a.runs, "max_comments": a.max_comments, "use_llm": False if a.no_llm else None}
    opts |= {s: not getattr(a, f"no_{s}") for s in ("intent", "evidence", "verifier", "consistency", "ranker")}
    res = review(pr, opts)
    _print(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
        print(f"\nwrote {a.json}")
    if a.post:
        from agent.github import post_review

        post_review(pr.repo, pr.number, pr.head_sha, res["comments"], res["summary_markdown"])
        print("posted the review on GitHub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
