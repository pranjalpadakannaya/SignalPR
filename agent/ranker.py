"""Step 5 of the review: risk ranker.

The risk model from issue #10 (trained on which files human reviewers actually commented on in
AIDev-pop) gives every changed file a probability of drawing a reviewer comment. Each verified
comment is scored by its file's risk, its severity and how strong its evidence is, and the
highest scores are shown first.

    score = 0.35 * file risk (relative to the riskiest file) + 0.45 * severity + 0.20 * evidence
"""

from __future__ import annotations

from agent.state import SEVERITY_WEIGHT, Comment, PRInput
from risk.predict import predict_risk

WEIGHTS = {"risk": 0.35, "severity": 0.45, "evidence": 0.20}
NO_RISK_WEIGHTS = {"risk": 0.0, "severity": 0.6, "evidence": 0.4}


def file_risk(pr: PRInput, model_path: str | None = None) -> list[dict]:
    """Per-file risk, highest first (falls back to change size when no model file is present)."""
    return predict_risk(pr.files, pr.meta(), model_path=model_path)


def evidence_strength(c: Comment) -> float:
    tool = 1.0 if "tool evidence" in c.verified_by else 0.0
    quote = 1.0 if "quote matches code" in c.verified_by else 0.0
    return min(1.0, 0.5 * tool + 0.25 * quote + 0.25 * c.support + (0.25 if tool and quote else 0.0))


def rank(comments: list[Comment], risks: list[dict], mode: str = "risk") -> list[Comment]:
    """mode: "risk" (SignalPR), "no_risk" (ablation), "size" (change-size baseline), "severity" (tools baseline)."""
    by_file = {r["file"]: r for r in risks}
    for c in comments:
        r = by_file.get(c.file, {"risk": 0.0, "risk_rel": 0.0, "additions": 0, "deletions": 0})
        c.risk = float(r["risk"])
        parts = {
            "risk": float(r["risk_rel"]) if mode == "risk" else 0.0,
            "severity": SEVERITY_WEIGHT[c.severity],
            "evidence": evidence_strength(c),
        }
        if mode == "size":
            c.score = float((r.get("additions") or 0) + (r.get("deletions") or 0))
        elif mode == "severity":
            c.score = parts["severity"] + 0.01 * parts["evidence"]
        else:
            w = WEIGHTS if mode == "risk" else NO_RISK_WEIGHTS
            c.score = round(sum(w[k] * parts[k] for k in w), 4)
        c.score_parts = {k: round(v, 3) for k, v in parts.items()}
    return sorted(comments, key=lambda c: (-c.score, c.file, c.line or 0))
