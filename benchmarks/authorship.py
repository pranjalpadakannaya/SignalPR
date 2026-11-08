"""Was a PR written by a coding agent or by a person? Used for the agent- vs human-written breakdown.

Neither benchmark labels this, so we infer it from the same kinds of signals AIDev uses: the PR
author account, the head branch name, markers that agents leave in the PR body, and commit
co-author trailers. Result: author_kind in {"agent", "human", "bot", "unknown"} plus which agent.
"bot" is dependency/automation bots (Dependabot, Renovate) that are not coding agents.
"""

from __future__ import annotations

import re
from typing import Any

# (agent name, regex) checked against each signal, most specific first
AUTHOR_LOGINS = [
    ("Copilot", r"^(copilot-swe-agent|copilot)(\[bot\])?$"),
    ("Devin", r"^devin-ai-integration(\[bot\])?$"),
    ("Cursor", r"^cursor(agent)?(\[bot\])?$"),
    ("Claude_Code", r"^claude(\[bot\])?$"),
    ("Jules", r"^google-labs-jules(\[bot\])?$"),
    ("OpenAI_Codex", r"^(chatgpt-codex-connector|codex)(\[bot\])?$"),
    ("Sweep", r"^sweep-ai(\[bot\])?$"),
]
BRANCH_PREFIXES = [
    ("OpenAI_Codex", r"^codex[/-]"),
    ("Copilot", r"^copilot/"),
    ("Cursor", r"^cursor/"),
    ("Devin", r"^devin/"),
    ("Claude_Code", r"^claude/"),
    ("Jules", r"^jules[/-]"),
]
TEXT_MARKERS = [
    ("Claude_Code", r"generated with \[?claude code|co-authored-by:[^\n]*(claude|noreply@anthropic\.com)"),
    ("Devin", r"app\.devin\.ai|devin run|requested by[^\n]*devin"),
    ("Cursor", r"cursor\.com/(agents|background-agent)|co-authored-by:[^\n]*cursoragent"),
    ("OpenAI_Codex", r"chatgpt\.com/codex|co-authored-by:[^\n]*codex"),
    ("Copilot", r"co-authored-by:[^\n]*copilot"),
    ("Jules", r"jules\.google"),
    ("Aider", r"co-authored-by:[^\n]*aider|^aider:"),
]
AUTOMATION_BOTS = r"^(dependabot|renovate|github-actions|pre-commit-ci|snyk-bot|greenkeeper)(\[bot\])?$"


def _first(rules: list[tuple[str, str]], text: str, flags: int = re.I | re.M) -> str | None:
    for name, pattern in rules:
        if text and re.search(pattern, text, flags):
            return name
    return None


def classify(pr: dict[str, Any] | None) -> dict[str, Any]:
    """`pr` is the dict from github.pull_request(); returns author, author_kind, agent, agent_signal."""
    if not pr:
        return {"author": None, "author_kind": "unknown", "agent": None, "agent_signal": None}
    author = pr.get("author") or {}
    login = author.get("login") or ""
    out = {"author": login or None, "author_kind": "human", "agent": None, "agent_signal": None}

    checks = [
        ("author", _first(AUTHOR_LOGINS, login)),
        ("branch", _first(BRANCH_PREFIXES, pr.get("headRefName") or "")),
        ("body", _first(TEXT_MARKERS, pr.get("body") or "")),
        ("commits", _first(TEXT_MARKERS, "\n".join(pr.get("commit_messages") or []))),
    ]
    for signal, agent in checks:
        if agent:
            out.update(author_kind="agent", agent=agent, agent_signal=signal)
            return out
    if re.match(AUTOMATION_BOTS, login, re.I) or (author.get("__typename") == "Bot"):
        out["author_kind"] = "bot"
    elif not login:
        out["author_kind"] = "unknown"
    return out
