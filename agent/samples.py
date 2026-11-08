"""Demo pull requests (agent/samples/*.json): a unified diff, PR/issue text, and the known issues ("gold")."""

from __future__ import annotations

import json
from pathlib import Path

from agent.state import PRInput
from tools.diff import parse_unified_diff

SAMPLES_DIR = Path(__file__).with_name("samples")


def pr_from_dict(d: dict) -> PRInput:
    return PRInput(
        files=parse_unified_diff(d.get("diff", "")),
        title=d.get("title", ""),
        description=d.get("description", ""),
        issue_title=d.get("issue_title", ""),
        issue_body=d.get("issue_body", ""),
        agent=d.get("agent", "unknown"),
        repo=d.get("repo", ""),
        repo_language=d.get("repo_language"),
        stars=int(d.get("stars") or 0),
        forks=int(d.get("forks") or 0),
        url=d.get("url", ""),
        repo_path=d.get("repo_path"),
        test_cmd=d.get("test_cmd"),
    )


def list_samples() -> list[dict]:
    out = []
    for p in sorted(SAMPLES_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        out.append(
            {
                "id": d["id"],
                "title": d["title"],
                "agent": d.get("agent"),
                "repo": d.get("repo"),
                "issue_title": d.get("issue_title", ""),
                "known_issues": len(d.get("gold", [])),
            }
        )
    return out


def load_sample(sample_id: str) -> dict:
    p = SAMPLES_DIR / f"{Path(sample_id).name}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no sample named {sample_id!r}; available: {[s['id'] for s in list_samples()]}")
    return json.loads(p.read_text())
