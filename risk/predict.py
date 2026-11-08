"""predict_risk(): the risk model's interface for the agent's ranker and the baselines (issue #10, task 6).

    from risk import predict_risk
    risks = predict_risk(
        files=[{"filename": "app/routes/search.py", "status": "added", "additions": 30, "deletions": 0,
                "patch": "@@ -0,0 +1,30 @@\\n+..."},           # patch is optional
               {"filename": "app/settings.py", "status": "modified", "additions": 4, "deletions": 2}],
        pr={"agent": "OpenAI_Codex", "title": "Add user search endpoint", "body": "...",
            "issue_text": "Admin page needs user search", "stars": 640, "forks": 88, "repo_language": "Python"})
    # -> [{"file": ..., "risk": 0.051, "risk_rel": 1.0, "rank": 1, ...}, ...]   sorted by risk, highest first

`risk` is the calibrated probability that a human reviewer leaves an inline comment on the file.
`risk_rel` is risk divided by the PR's highest risk (0..1), convenient for score mixing.
Files can also be tools.diff.FileDiff objects (from a parsed unified diff).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from risk.features import build_features
from risk.model import RiskModel
from risk.paths import MODEL_PATH

AIDEV_AGENTS = ("Claude_Code", "Copilot", "OpenAI_Codex", "Cursor", "Devin", "Google_Jules")


@lru_cache(maxsize=4)
def load_model(path: str | None = None) -> RiskModel | None:
    return RiskModel.load(path or MODEL_PATH)


def _file_row(f) -> dict:
    if hasattr(f, "hunks"):  # FileDiff
        return {
            "filename": f.path,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": f.patch_text(),
        }
    d = dict(f)
    d.setdefault("filename", d.pop("path", d.get("file")))
    return d


def predict_risk(files: list, pr: dict | None = None, model_path: str | None = None) -> list[dict]:
    if not files:
        return []
    pr = pr or {}
    rows = []
    for f in files:
        r = _file_row(f)
        r.setdefault("status", "modified")
        r.setdefault("additions", 0)
        r.setdefault("deletions", 0)
        r["changes"] = r.get("changes", (r["additions"] or 0) + (r["deletions"] or 0))
        rows.append(
            {
                "pr_id": 0,
                **r,
                "agent": pr.get("agent") if pr.get("agent") in AIDEV_AGENTS else "unknown",
                "title": pr.get("title", ""),
                "body": pr.get("body", "") or "",
                "body_len": len(pr.get("body", "") or ""),
                "issue_text": pr.get("issue_text", "") or "",
                "has_issue": int(bool(pr.get("issue_text"))),
                "stars": pr.get("stars", 0),
                "forks": pr.get("forks", 0),
                "repo_language": pr.get("repo_language"),
            }
        )
    df = pd.DataFrame(rows)
    if "patch" not in df.columns:
        df["patch"] = None
    feats = build_features(df)
    m = load_model(model_path)
    if m is not None:
        p = m.predict(feats)
        source = f"risk model ({m.backend}, PR-AUC {m.meta.get('test_pr_auc', float('nan')):.3f})"
    else:  # no trained model on disk: fall back to change size (the proposal's size baseline)
        p = 1 - np.exp(-feats["changes"].values / 150.0)
        source = "change-size fallback (no trained model found)"
    pmax = float(np.max(p)) or 1.0
    out = [
        {
            "file": fn,
            "risk": float(pi),
            "risk_rel": float(pi) / pmax if pmax > 0 else 0.0,
            "additions": int(a),
            "deletions": int(dl),
            "lang": lang,
            "is_test": bool(t),
            "source": source,
        }
        for fn, pi, a, dl, lang, t in zip(
            feats["filename"], p, feats["additions"], feats["deletions"], feats["lang"], feats["is_test"]
        )
    ]
    out.sort(key=lambda r: -r["risk"])
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out
