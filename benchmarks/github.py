"""Minimal GitHub client used by the benchmark downloaders (stdlib only, no extra installs).

Token: $GITHUB_TOKEN or $GH_TOKEN, else `gh auth token` if the GitHub CLI is logged in.
Unauthenticated access works for raw files but the API is limited to 60 requests/hour.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
UA = "SignalPR-benchmarks"


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub {status}: {message}")
        self.status = status


@lru_cache(maxsize=1)
def token() -> str | None:
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    if shutil.which("gh"):
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


def _request(url: str, accept: str = "application/vnd.github+json", data: bytes | None = None) -> bytes:
    headers = {"Accept": accept, "User-Agent": UA, "X-GitHub-Api-Version": "2022-11-28"}
    if token() and url.startswith(API):
        headers["Authorization"] = f"Bearer {token()}"
    for attempt in range(5):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            retry = e.code in (502, 503, 504) or (e.code in (403, 429) and "rate limit" in body.lower())
            if not retry or attempt == 4:
                raise GitHubError(e.code, body) from None
            reset = e.headers.get("x-ratelimit-reset")
            wait = max(1.0, float(reset) - time.time()) if reset and e.code in (403, 429) else 2.0**attempt
            time.sleep(min(wait, 900))
        except urllib.error.URLError:
            if attempt == 4:
                raise
            time.sleep(2.0**attempt)
    raise AssertionError("unreachable")


def raw_file(repo: str, commit: str, path: str) -> bytes:
    return _request(f"{RAW}/{repo}/{commit}/{path}", accept="*/*")


def rest(path: str, accept: str = "application/vnd.github+json") -> Any:
    body = _request(f"{API}/{path.lstrip('/')}", accept=accept)
    return json.loads(body) if accept.endswith("json") else body.decode("utf-8", "replace")


def graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    if not token():
        raise GitHubError(401, "GraphQL needs a token: set GITHUB_TOKEN or run `gh auth login`")
    payload = json.dumps({"query": query, "variables": variables}).encode()
    out = json.loads(_request(f"{API}/graphql", data=payload))
    if out.get("errors") and not out.get("data"):
        raise GitHubError(400, json.dumps(out["errors"])[:500])
    return out.get("data") or {}


def compare_diff(repo: str, base: str, head: str) -> str | None:
    """Unified diff base...head from the API, or None if GitHub refuses (too large, missing commit)."""
    try:
        return rest(f"repos/{repo}/compare/{base}...{head}", accept="application/vnd.github.diff")
    except GitHubError as e:
        if e.status in (404, 406, 422, 500):
            return None
        raise


PR_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    primaryLanguage { name }
    pullRequest(number: $number) {
      title body headRefName baseRefOid headRefOid
      additions deletions changedFiles createdAt
      author { login __typename }
      closingIssuesReferences(first: 10) { nodes { number title body url } }
      commits(first: 100) { nodes { commit { message author { name email } } } }
    }
  }
}
"""


def pull_request(repo: str, number: int) -> dict[str, Any] | None:
    """PR metadata (author, refs, linked issues, commit messages) or None if it no longer exists."""
    owner, name = repo.split("/", 1)
    try:
        data = graphql(PR_QUERY, {"owner": owner, "name": name, "number": int(number)})
    except GitHubError as e:
        if e.status in (401, 403):
            raise
        return None
    r = data.get("repository") or {}
    pr = r.get("pullRequest")
    if pr is None:
        return None
    pr["repo_language"] = (r.get("primaryLanguage") or {}).get("name")
    pr["linked_issues"] = (pr.pop("closingIssuesReferences") or {}).get("nodes") or []
    pr["commit_messages"] = [n["commit"]["message"] for n in (pr.pop("commits") or {}).get("nodes") or []]
    return pr


def issue(repo: str, number: int) -> dict[str, Any] | None:
    try:
        d = rest(f"repos/{repo}/issues/{number}")
    except GitHubError:
        return None
    if "pull_request" in d:  # "fixes #N" sometimes points at another PR
        return None
    return {"number": d["number"], "title": d.get("title") or "", "body": d.get("body") or "", "url": d["html_url"]}
