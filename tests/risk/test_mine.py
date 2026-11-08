"""End-to-end: tiny AIDev-shaped CSV tables -> mine() -> labelled, split, featurized files."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _tables(raw: Path):
    raw.mkdir(parents=True)
    prs, files, rc, rv, repos = [], [], [], [], []
    for r in range(20):
        repos.append(
            {
                "id": r,
                "url": f"https://api.github.com/repos/o/r{r}",
                "full_name": f"o/r{r}",
                "language": "Python",
                "stars": 200 + r,
                "forks": 10,
            }
        )
        for k in range(5):
            pid = r * 100 + k
            prs.append(
                {
                    "id": pid,
                    "number": k,
                    "title": f"feat: add thing {k}",
                    "body": "x",
                    "agent": "Copilot",
                    "user": "dev",
                    "state": "closed",
                    "created_at": f"2026-0{1 + k % 3}-1{k}T00:00:00Z",
                    "closed_at": "2026-04-01T00:00:00Z",
                    "merged_at": "2026-04-01T00:00:00Z",
                    "repo_id": r,
                    "repo_url": f"https://api.github.com/repos/o/r{r}",
                }
            )
            rv.append({"pr_id": pid, "user": "rev", "user_type": "User", "state": "COMMENTED"})
            for j, fn in enumerate(["src/core.py", "README.md", "tests/test_core.py"]):
                files.append(
                    {
                        "pr_id": pid,
                        "filename": fn,
                        "status": "modified",
                        "additions": 50 - 20 * j,
                        "deletions": 5,
                        "changes": 55 - 20 * j,
                    }
                )
            url = f"https://api.github.com/repos/o/r{r}/pulls/{k}"
            rc.append(
                {
                    "user": "rev",
                    "user_type": "User",
                    "path": "src/core.py",
                    "pull_request_url": url,
                    "body": "please handle None",
                    "created_at": "2026-04-01",
                }
            )
            rc.append(
                {
                    "user": "copilot[bot]",
                    "user_type": "Bot",
                    "path": "README.md",
                    "pull_request_url": url,
                    "body": "bot says hi",
                    "created_at": "2026-04-01",
                }
            )
            rc.append(
                {
                    "user": "rev",
                    "user_type": "User",
                    "path": "tests/test_core.py",
                    "pull_request_url": url,
                    "body": "   ",
                    "created_at": "2026-04-01",
                }
            )  # empty -> dropped
    for name, rows in [
        ("pull_request", prs),
        ("pr_files", files),
        ("pr_review_comments", rc),
        ("pr_reviews", rv),
        ("repository", repos),
    ]:
        pd.DataFrame(rows).to_csv(raw / f"{name}.csv.gz", index=False)


def test_mine_end_to_end(tmp_path):
    _tables(tmp_path / "data" / "raw")
    env = {
        **os.environ,
        "SIGNALPR_DATA_DIR": str(tmp_path / "data"),
        "SIGNALPR_RISK_RESULTS": str(tmp_path / "res"),
        "PYTHONPATH": str(ROOT),
    }
    subprocess.run([sys.executable, "-m", "risk", "mine"], check=True, env=env, capture_output=True)
    df = pd.read_pickle(tmp_path / "data" / "processed" / "files.pkl")
    assert len(df) == 300
    lab = df.groupby("filename")["label"].mean()
    assert lab["src/core.py"] == 1.0  # human comment
    assert lab["README.md"] == 0.0  # bot comment ignored
    assert lab["tests/test_core.py"] == 0.0  # empty comment ignored
    # repo-disjoint split
    repos = {s: set(df.loc[df["split"] == s, "repo_id"]) for s in ("train", "val", "test")}
    assert not (repos["train"] & repos["test"]) and not (repos["train"] & repos["val"])
    stats = json.loads((tmp_path / "res" / "dataset_stats.json").read_text())
    assert stats["totals"]["inline_comments_human"] == 100
