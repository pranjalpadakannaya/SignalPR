"""The SignalPR review pipeline:  intent -> evidence -> specialists -> verifier -> ranker.

Built as a LangGraph StateGraph when `langgraph` is installed (as in the proposal); otherwise the
same five node functions run in order, so results are identical either way.

    from agent.pipeline import review
    from agent.state import PRInput
    result = review(PRInput.from_diff(open("change.diff").read(), title="Add search"), {"mode": "signalpr"})

`mode` selects SignalPR or a baseline (every system uses the same LLM and the same input):
  signalpr        full pipeline
  single_prompt   one prompt to the same model; no tools, no verifier, no ranker
  tools_only      static-analysis findings turned into comments, ranked by severity
  size_rank       SignalPR's verified comments, ranked by file change size instead of risk
Steps can be switched off one at a time for the ablation study (intent, evidence, verifier,
consistency, ranker).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, TypedDict

from agent import evidence as evidence_step
from agent import intent as intent_step
from agent import llm, ranker, specialists, verifier
from agent.state import SEVERITIES, Comment, PRInput

MODES = ("signalpr", "single_prompt", "tools_only", "size_rank")


@dataclass
class Options:
    mode: str = "signalpr"
    use_llm: bool | None = None  # None = use it if the server is reachable
    runs: int = 3  # self-consistency samples per specialist
    intent: bool = True
    evidence: bool = True
    verifier: bool = True
    consistency: bool = True
    ranker: bool = True
    run_tests: bool = True
    max_comments: int = 8
    specialists: list = field(default_factory=lambda: list(specialists.SPECIALISTS))

    @classmethod
    def from_dict(cls, d: dict | None) -> "Options":
        names = {f.name for f in fields(cls)}
        o = cls(**{k: v for k, v in (d or {}).items() if k in names and v is not None})
        if o.mode not in MODES:
            raise ValueError(f"unknown mode {o.mode!r}; choose from {MODES}")
        o.runs = max(1, min(int(o.runs), 7))
        return o


class ReviewState(TypedDict, total=False):
    pr: PRInput
    options: Options
    use_llm: bool
    intent: dict
    evidence: dict
    candidates: list
    runs: int
    errors: list
    kept: list
    dropped: list
    risks: list
    ranked: list
    trace: list


# ------------------------------------------------------------------ nodes
def node_intent(s: ReviewState) -> dict:
    o = s["options"]
    if not o.intent:
        return {"intent": {"mode": "off", "requirements": [], "unrequested": [], "summary": "intent check disabled"}}
    return {"intent": intent_step.check_intent(s["pr"], s["use_llm"])}


def node_evidence(s: ReviewState) -> dict:
    o = s["options"]
    if not o.evidence:
        return {"evidence": {"items": [], "tools": {}, "usage": {}}}
    return {"evidence": evidence_step.collect(s["pr"], run_tests=o.run_tests)}


def node_specialists(s: ReviewState) -> dict:
    o, pr, ev = s["options"], s["pr"], s["evidence"]["items"]
    if o.mode == "tools_only":
        cands = [c for n in o.specialists for c in specialists.tool_comments(n, pr, ev, s["intent"])]
        return {"candidates": cands, "runs": 1, "errors": []}
    res = specialists.review(pr, ev, s["intent"], s["use_llm"], o.runs, o.specialists)
    return {"candidates": res["candidates"], "runs": res["runs"], "errors": res["errors"]}


def node_verifier(s: ReviewState) -> dict:
    o, pr = s["options"], s["pr"]
    if o.verifier and o.mode != "tools_only":
        res = verifier.verify(pr, s["candidates"], s["evidence"]["items"], s["runs"], o.consistency)
        return {"kept": res["kept"], "dropped": res["dropped"]}
    kept, dropped = [], []
    for c in s["candidates"]:  # without the verifier only the line anchor is checked
        f = pr.file(c.file)
        if f is not None and c.line:
            c.file, c.snippet, c.support = f.path, f.snippet(c.line, 2), 1.0
            c.verified_by = ["tool evidence"] if c.evidence else []
            kept.append(c)
        else:
            c.drop_reason = "no file/line in the diff"
            dropped.append(c)
    return {"kept": kept, "dropped": dropped}


def node_ranker(s: ReviewState) -> dict:
    o = s["options"]
    risks = ranker.file_risk(s["pr"])
    mode = {"size_rank": "size", "tools_only": "severity"}.get(o.mode, "risk" if o.ranker else "no_risk")
    return {"risks": risks, "ranked": ranker.rank(s["kept"], risks, mode=mode)}


NODES: list[tuple[str, Callable[[ReviewState], dict]]] = [
    ("intent", node_intent),
    ("evidence", node_evidence),
    ("specialists", node_specialists),
    ("verifier", node_verifier),
    ("ranker", node_ranker),
]


DETAILS: dict[str, Callable[[dict], str]] = {
    "intent": lambda o: o["intent"].get("summary", ""),
    "evidence": lambda o: f"{len(o['evidence']['items'])} findings",
    "specialists": lambda o: (
        f"{len(o['candidates'])} candidates" + (f", {len(o['errors'])} LLM errors" if o["errors"] else "")
    ),
    "verifier": lambda o: f"kept {len(o['kept'])}, filtered {len(o['dropped'])}",
    "ranker": lambda o: o["risks"][0]["source"] if o["risks"] else "no files",
}


def _timed(name, fn, progress):
    def run(s: ReviewState) -> dict:
        if progress:
            progress(name, "running")
        t = time.time()
        out = fn(s)
        step = {"step": name, "ms": int((time.time() - t) * 1000), "detail": DETAILS[name](out)}
        out["trace"] = list(s.get("trace", [])) + [step]
        if progress:
            progress(name, "done")
        return out

    return run


def build_graph(progress=None):
    """Compile the LangGraph graph, or return None if langgraph is not installed."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None
    g = StateGraph(ReviewState)
    for name, fn in NODES:
        g.add_node(name, _timed(name, fn, progress))
    g.set_entry_point(NODES[0][0])
    for (a, _), (b, _) in zip(NODES, NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(NODES[-1][0], END)
    return g.compile()


def _run_nodes(state: ReviewState, progress=None) -> ReviewState:
    graph = build_graph(progress)
    if graph is not None:
        return graph.invoke(state)
    for name, fn in NODES:
        state.update(_timed(name, fn, progress)(state))
    return state


# ------------------------------------------------------------------ single-prompt baseline
SINGLE_PROMPT = """You are an expert code reviewer. Review this pull request and list the problems you find.
Reply with JSON only: {"comments":[{"file":"<path>","line":<int>,"severity":"critical|medium|low",
"title":"<short>","body":"<explanation>","suggestion":"<fix>","quote":"<exact line>"}]}"""


def single_prompt_comments(pr: PRInput) -> list[Comment]:
    diff = "\n\n".join(f"### {f.path}\n{f.numbered_patch(250)}" for f in pr.files)[:18000]
    user = (
        f"PR: {pr.title}\n{pr.description[:1500]}\n\nISSUE: {pr.issue_title}\n{pr.issue_body[:1500]}\n\nDIFF:\n{diff}"
    )
    data, _ = llm.chat_json(SINGLE_PROMPT, user, temperature=0.2, tag="baseline/single_prompt")
    out = []
    for c in (data or {}).get("comments", []) if isinstance(data, dict) else []:
        if not isinstance(c, dict) or not c.get("file"):
            continue
        try:
            line = int(c.get("line"))
        except (TypeError, ValueError):
            line = None
        f = pr.file(str(c["file"]))
        out.append(
            Comment(
                file=f.path if f else str(c["file"]),
                line=line,
                category="correctness",
                severity=c.get("severity") if c.get("severity") in SEVERITIES else "medium",
                title=str(c.get("title", ""))[:140],
                body=str(c.get("body", ""))[:1500],
                source="single_prompt/llm",
                quote=str(c.get("quote", ""))[:300],
                suggestion=str(c.get("suggestion", ""))[:500],
                support=1.0,
                snippet=f.snippet(line, 2) if f and line else [],
            )
        )
    return out


# ------------------------------------------------------------------ entry point
def review(pr: PRInput, options: Options | dict | None = None, progress=None) -> dict:
    o = options if isinstance(options, Options) else Options.from_dict(options)
    t0 = time.time()
    st = llm.status()
    use_llm = st["available"] if o.use_llm is None else bool(o.use_llm and st["available"])
    if o.use_llm and not st["available"]:
        raise RuntimeError(f"LLM requested but not reachable: {st['reason']}")

    if o.mode == "single_prompt":
        if not use_llm:
            raise RuntimeError("The single-prompt baseline needs an LLM server (see scripts/serve/README.md).")
        t = time.time()
        comments = single_prompt_comments(pr)
        risks = ranker.file_risk(pr)
        state: ReviewState = {
            "pr": pr,
            "options": o,
            "use_llm": True,
            "ranked": comments,
            "dropped": [],
            "risks": risks,
            "errors": [],
            "intent": {"mode": "off", "requirements": [], "unrequested": []},
            "evidence": {"items": [], "tools": {}, "usage": {}},
            "trace": [
                {"step": "single prompt", "ms": int((time.time() - t) * 1000), "detail": f"{len(comments)} comments"}
            ],
        }
    else:
        state = _run_nodes({"pr": pr, "options": o, "use_llm": use_llm, "trace": []}, progress)
    return to_result(state, st, time.time() - t0)


def comment_markdown(c: Comment) -> str:
    label = {"critical": "Critical", "medium": "Medium", "low": "Low"}[c.severity]
    md = f"**[{label}] {c.title}**\n\n{c.body}"
    if c.suggestion:
        md += f"\n\n**Suggestion:** {c.suggestion}"
    return md + (
        f"\n\n<sub>SignalPR · {c.category} · evidence: {', '.join(c.verified_by) or 'n/a'} · "
        f"file risk {c.risk:.1%}</sub>"
    )


def to_result(s: ReviewState, llm_status: dict, seconds: float) -> dict[str, Any]:
    pr, o = s["pr"], s["options"]
    ranked = s.get("ranked", [])
    shown, more = ranked[: o.max_comments], ranked[o.max_comments :]
    intent = s.get("intent", {})
    ev = s.get("evidence", {"items": [], "tools": {}, "usage": {}})
    dropped = s.get("dropped", [])
    summary = (
        f"### SignalPR review\n**{len(shown)}** comment(s)"
        + (f" (+{len(more)} lower-priority)" if more else "")
        + f", **{len(dropped)}** candidate(s) filtered by the verifier.\n\n"
        + (f"**Intent:** {intent.get('summary', '')}\n\n" if intent.get("summary") else "")
        + "\n".join(f"- [{r['status']}] {r['text']}" for r in intent.get("requirements", [])[:8])
    )
    as_dict = lambda c: c.to_dict() | {"markdown": comment_markdown(c)}  # noqa: E731
    return {
        "pr": {
            "title": pr.title,
            "url": pr.url,
            "repo": pr.repo,
            "number": pr.number,
            "agent": pr.agent,
            "files": len(pr.files),
            "additions": sum(f.additions for f in pr.files),
            "deletions": sum(f.deletions for f in pr.files),
            "has_issue": pr.has_issue,
        },
        "options": {f.name: getattr(o, f.name) for f in fields(o)} | {"llm": s.get("use_llm", False)},
        "llm": {
            "used": s.get("use_llm", False),
            "model": llm_status.get("model"),
            "available": llm_status["available"],
            "reason": llm_status.get("reason", ""),
            "errors": s.get("errors", []),
        },
        "intent": intent,
        "evidence": [e.to_dict() for e in ev["items"]],
        "tools": ev["tools"],
        "usage": ev.get("usage", {}),
        "file_risk": s.get("risks", []),
        "comments": [as_dict(c) for c in shown],
        "more_comments": [as_dict(c) for c in more],
        "dropped": [c.to_dict() for c in dropped],
        "stats": {
            "shown": len(shown),
            "more": len(more),
            "dropped": len(dropped),
            "evidence": len(ev["items"]),
            "severity": {k: sum(c.severity == k for c in ranked) for k in SEVERITIES},
            "seconds": round(seconds, 2),
        },
        "summary_markdown": summary,
        "trace": s.get("trace", []),
    }
