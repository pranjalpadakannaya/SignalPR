"""GitHub REST access for the reviewer: fetch a pull request (files, patches, full file contents,
linked issue, repo metadata) and post the review back as inline comments.

Standard library only. Set GITHUB_TOKEN (a free personal access token) for private repositories
and higher rate limits; the GitHub App path gets an installation token instead (app/webhook.py).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from agent.state import PRInput
from tools.diff import FileDiff, parse_hunks

API = os.environ.get("GITHUB_API", "https://api.github.com")
PR_URL = re.compile(r"github\.com/([^/\s]+)/([^/\s]+)/pull/(\d+)")
ISSUE_REF = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b|(?<![\w/])#(\d+)\b", re.I)


class GitHubError(RuntimeError):
    pass


def _request(method: str, path: str, token: str | None = None, body: dict | None = None, **params):
    url = API.rstrip("/") + path + (("?" + urllib.parse.urlencode(params)) if params else "")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "SignalPR", "X-GitHub-Api-Version": "2022-11-28"}
    tok = token or os.environ.get("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed https API host
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:300]
        if e.code == 403 and "rate limit" in text.lower():
            raise GitHubError("GitHub API rate limit hit; set GITHUB_TOKEN and retry.") from e
        if e.code == 404:
            raise GitHubError(f"GitHub returned 404 for {path} (private repo? set GITHUB_TOKEN).") from e
        raise GitHubError(f"GitHub {e.code} for {path}: {text}") from e
    except urllib.error.URLError as e:
        raise GitHubError(f"cannot reach GitHub: {e.reason}") from e


def detect_agent(pr: dict, commit_text: str = "") -> str:
    """Guess which coding agent opened the PR (same labels as AIDev)."""
    login = ((pr.get("user") or {}).get("login") or "").lower()
    branch = ((pr.get("head") or {}).get("ref") or "").lower()
    body = ((pr.get("body") or "") + " " + commit_text).lower()
    if login.startswith("copilot") or branch.startswith("copilot/"):
        return "Copilot"
    if "devin-ai-integration" in login or branch.startswith("devin/"):
        return "Devin"
    if branch.startswith("codex/") or "chatgpt.com/codex" in body:
        return "OpenAI_Codex"
    if branch.startswith("cursor/") or "cursor agent" in body:
        return "Cursor"
    if branch.startswith("claude/") or "claude.com/claude-code" in body:
        return "Claude_Code"
    if "jules" in login or branch.startswith("jules"):
        return "Google_Jules"
    return "unknown"


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_URL.search(url or "")
    if not m:
        raise ValueError("expected a URL like https://github.com/owner/repo/pull/123")
    return m.group(1), m.group(2), int(m.group(3))


def fetch_pr(url: str, token: str | None = None, with_content: bool = True, max_files: int = 60) -> PRInput:
    owner, repo, num = parse_pr_url(url)
    base = f"/repos/{owner}/{repo}"
    pr = _request("GET", f"{base}/pulls/{num}", token)
    meta = _request("GET", base, token)
    files_json, page = [], 1
    while len(files_json) < max_files:
        batch = _request("GET", f"{base}/pulls/{num}/files", token, per_page=100, page=page)
        files_json += batch
        if len(batch) < 100:
            break
        page += 1
    head = pr["head"]["sha"]
    files = []
    for fj in files_json[:max_files]:
        status = {"changed": "modified"}.get(fj["status"], fj["status"])
        f = FileDiff(
            path=fj["filename"],
            old_path=fj.get("previous_filename"),
            status=status,
            hunks=parse_hunks(fj.get("patch") or ""),
            binary=not fj.get("patch") and fj.get("changes", 0) > 0,
        )
        if with_content and status != "removed" and not f.binary and fj.get("changes", 0) < 3000:
            try:
                c = _request("GET", f"{base}/contents/{urllib.parse.quote(f.path)}", token, ref=head)
                if isinstance(c, dict) and c.get("encoding") == "base64" and c.get("size", 0) < 400_000:
                    f.content = base64.b64decode(c["content"]).decode("utf-8", errors="replace")
            except GitHubError:
                pass
        files.append(f)
    issue_title = issue_body = ""
    for a, b in ISSUE_REF.findall(pr.get("body") or ""):
        ref = int(a or b)
        if ref == num:
            continue
        try:
            iss = _request("GET", f"{base}/issues/{ref}", token)
        except GitHubError:
            continue
        if "pull_request" not in iss:
            issue_title, issue_body = iss.get("title", ""), iss.get("body") or ""
            break
    return PRInput(
        files=files,
        title=pr.get("title", ""),
        description=pr.get("body") or "",
        issue_title=issue_title,
        issue_body=issue_body,
        agent=detect_agent(pr),
        repo=f"{owner}/{repo}",
        repo_language=meta.get("language"),
        stars=meta.get("stargazers_count", 0),
        forks=meta.get("forks_count", 0),
        url=pr.get("html_url", url),
        number=num,
        head_sha=head,
    )


def post_review(
    repo: str, number: int, head_sha: str, comments: list[dict], summary: str, token: str | None = None
) -> dict:
    """Post one review (event=COMMENT) with an inline comment per ranked comment."""
    body = {
        "commit_id": head_sha,
        "event": "COMMENT",
        "body": summary,
        "comments": [
            {"path": c["file"], "line": c["line"], "side": "RIGHT", "body": c["markdown"]}
            for c in comments
            if c.get("line")
        ],
    }
    path = f"/repos/{repo}/pulls/{number}/reviews"
    try:
        return _request("POST", path, token, body)
    except GitHubError as e:
        if "422" not in str(e) or not body["comments"]:
            raise
        # a line GitHub cannot anchor: post everything in the review body instead
        body["body"] += "\n\n" + "\n".join(
            f"- `{c['path']}:{c['line']}` {c['body'].splitlines()[0]}" for c in body["comments"]
        )
        body["comments"] = []
        return _request("POST", path, token, body)


def app_installation_token(installation_id: int, app_id: str | None = None, key_path: str | None = None) -> str:
    """Exchange GitHub App credentials for a short-lived installation token (needs PyJWT + cryptography)."""
    import jwt  # PyJWT, only needed for the GitHub App

    app_id = app_id or os.environ["GITHUB_APP_ID"]
    with open(key_path or os.environ["GITHUB_APP_PRIVATE_KEY_PATH"]) as fh:
        key = fh.read()
    now = int(time.time())
    app_jwt = jwt.encode({"iat": now - 60, "exp": now + 540, "iss": app_id}, key, algorithm="RS256")
    return _request("POST", f"/app/installations/{installation_id}/access_tokens", app_jwt)["token"]
