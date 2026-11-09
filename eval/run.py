"""Evaluation harness (issue #7): SignalPR, the baselines and the ablations on the same PRs and model.

Benchmarks come from benchmarks.load_benchmark() in the PRExample format (#4, #5):
    martian       DEV set: tune here (loaded with purpose="tune")
    reviewbench   TEST set: frozen config only (loaded with purpose="eval")
    demo          the four demo PRs in agent/samples/ (harness check, not a result)
    <file.jsonl>  any file of PRExample lines
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent import llm
from agent.pipeline import Options, review
from agent.samples import SAMPLES_DIR, pr_from_dict
from agent.state import PRInput
from benchmarks import load_benchmark
from benchmarks.schema import PRExample, read_jsonl
from eval.judge import llm_match
from eval.metrics import aggregate, pr_metrics

SYSTEMS = {
    "SignalPR (full)": {"mode": "signalpr"},
    "Single-prompt LLM (same model)": {"mode": "single_prompt", "_needs_llm": True},
    "Code-analysis tools only": {"mode": "tools_only"},
    "SignalPR, ranked by change size": {"mode": "size_rank"},
    "– intent checker": {"intent": False},
    "– evidence collector": {"evidence": False},
    "– verifier": {"verifier": False},
    "– self-consistency": {"consistency": False, "_needs_llm": True},
    "– risk ranker": {"ranker": False},
}
PURPOSE = {"martian": "tune", "reviewbench": "eval"}


def pr_input(ex: PRExample) -> PRInput:
    issue = ex.linked_issues[0] if ex.linked_issues else None
    return PRInput.from_diff(
        ex.diff,
        title=ex.title,
        description=ex.description or "",
        issue_title=issue.title if issue else "",
        issue_body=ex.linked_issue_text if issue else "",
        agent=ex.metadata.get("agent") or "unknown",
        repo=ex.repo,
        repo_language=ex.metadata.get("language"),
        url=ex.pr_url,
        number=ex.pr_number,
        head_sha=ex.head_sha,
    )


def known_issues(ex: PRExample, profile: str | None = None) -> list[dict]:
    out = []
    for k in ex.known_issues:
        if profile and "profiles" in k.extra and profile not in k.extra["profiles"]:
            continue
        out.append(
            {
                "id": k.id,
                "body": k.body,
                "severity": k.severity,
                "category": k.category,
                "file": k.file,
                "start_line": k.start_line,
                "end_line": k.end_line,
                "label": k.label,
            }
        )
    return out


def load_items(bench: str, subset: str | None = None, profile: str | None = None) -> list[dict]:
    """[{id, pr: PRInput, issues: [known issue dicts], meta: {...}}] for any benchmark name or file."""
    if bench == "demo":
        rows = [json.loads(p.read_text()) for p in sorted(SAMPLES_DIR.glob("*.json"))]
        return [
            {
                "id": d["id"],
                "pr": pr_from_dict(d),
                "meta": {},
                "issues": [
                    {
                        "id": f"{d['id']}#{i}",
                        "body": g["description"],
                        "severity": g["severity"],
                        "category": g["type"],
                        "file": g["file"],
                        "start_line": g["line"],
                        "end_line": g["line"],
                        "label": "tp",
                    }
                    for i, g in enumerate(d.get("gold", []))
                ],
            }
            for d in rows
        ]
    if bench in PURPOSE:
        examples = load_benchmark(bench, purpose=PURPOSE[bench], subset=subset)
    else:
        examples = read_jsonl(Path(bench))
    return [
        {
            "id": ex.id,
            "pr": pr_input(ex),
            "issues": known_issues(ex, profile),
            "meta": {k: ex.metadata.get(k) for k in ("language", "size_bucket", "author_kind")},
        }
        for ex in examples
    ]


def run_benchmark(
    bench: str = "demo",
    systems: list[str] | None = None,
    repeats: int = 1,
    llm_runs: int = 3,
    out: str | Path | None = None,
    limit: int | None = None,
    progress=None,
    verbose: bool = False,
    judge: str = "auto",
    subset: str | None = None,
    profile: str | None = None,
) -> dict:
    items = load_items(bench, subset, profile)[:limit]
    st = llm.status(force=True)
    has_llm = st["available"]
    unlocated = any(not g.get("file") for it in items for g in it["issues"])
    if judge == "auto":
        judge = "llm" if unlocated and has_llm else "lexical"
    if judge == "llm" and not has_llm:
        raise RuntimeError("--judge llm needs a reachable LLM server")
    res = {
        "benchmark": bench,
        "subset": subset,
        "profile": profile,
        "n_prs": len(items),
        "n_known_issues": sum(sum(g["label"] == "tp" for g in it["issues"]) for it in items),
        "llm": {"model": st.get("model")} if has_llm else None,
        "judge": judge,
        "repeats": repeats,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "systems": {},
        "skipped": {},
    }
    if judge == "lexical" and unlocated:
        res["warning"] = "known issues without a file/line are matched on wording only; use --judge llm"
    for name in systems or list(SYSTEMS):
        cfg = dict(SYSTEMS[name])
        if cfg.pop("_needs_llm", False) and not has_llm:
            res["skipped"][name] = "needs an LLM (none reachable)"
            continue
        cfg.setdefault("runs", llm_runs)
        rows, t0 = [], time.time()
        for rep in range(repeats):
            for j, it in enumerate(items):
                if progress:
                    progress(f"{name}: PR {j + 1}/{len(items)}" + (f" (repeat {rep + 1})" if repeats > 1 else ""))
                r = review(it["pr"], Options.from_dict(cfg))
                shown = r["comments"] + r["more_comments"]
                matches = llm_match(shown, it["issues"]) if judge == "llm" else None
                rows.append(pr_metrics(shown, it["issues"], matches) | {"id": it["id"], "rep": rep, **it["meta"]})
        res["systems"][name] = aggregate(rows) | {"seconds": round(time.time() - t0, 1), "per_pr": rows}
        if verbose:
            a = res["systems"][name]
            cpi = a["comments_per_real_issue"]
            print(
                f"{name:34s} P={a['precision']:.2f} R={a['recall']:.2f} F1={a['f1']:.2f} R@3={a['recall@3']:.2f} "
                f"sevR={a['sev_weighted_recall']:.2f} comments/issue={cpi if cpi is None else round(cpi, 2)}"
            )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


def to_markdown(res: dict) -> str:
    lines = [
        f"# Evaluation: {res['benchmark']}" + (f" ({res['subset']})" if res.get("subset") else ""),
        "",
        f"{res['n_prs']} PRs · {res['n_known_issues']} real known issues · "
        f"{'LLM ' + res['llm']['model'] if res['llm'] else 'no LLM (tool evidence only)'} · judge {res['judge']} · "
        f"{res['repeats']} repeat(s) · {res['created']}",
        "",
        "| System | Precision | Recall | F1 (95% CI) | Sev-weighted recall | Recall@3 | Comments / real issue |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, s in res["systems"].items():
        ci = s.get("ci95", {}).get("f1")
        cpi = s["comments_per_real_issue"]
        lines.append(
            f"| {k} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f}"
            + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "")
            + f" | {s['sev_weighted_recall']:.3f} | {s['recall@3']:.3f} | {'–' if cpi is None else f'{cpi:.2f}'} |"
        )
    if res["skipped"]:
        lines += ["", "Skipped: " + "; ".join(f"{k} ({v})" for k, v in res["skipped"].items())]
    if res.get("warning"):
        lines += ["", f"Note: {res['warning']}"]
    return "\n".join(lines) + "\n"
