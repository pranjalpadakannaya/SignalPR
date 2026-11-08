"""Check out a benchmark PR's repository at its base or head commit (needed for tests and tools).

Each source repo is fetched once into a shared bare cache (repos/.cache/<owner>_<repo>.git) that
holds only the base and head commits (shallow, depth 1, no history) instead of a full clone. Each PR then gets a cheap `git worktree` at
repos/<benchmark>/<pr_id>/ (head by default). File contents are downloaded lazily on checkout,
so even very large repos (Sentry, Grafana) stay manageable.

    python -m benchmarks checkout martian                     # every Martian PR at head
    python -m benchmarks checkout martian --id martian/getsentry_sentry_93824 --ref base
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from benchmarks.registry import REPOS_DIR
from benchmarks.schema import PRExample

CACHE_DIR = REPOS_DIR / ".cache"


class CheckoutError(RuntimeError):
    pass


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if check and proc.returncode != 0:
        raise CheckoutError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:500]}")
    return proc


def _slug(clone_url: str) -> str:
    m = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$", clone_url)
    return f"{m.group(1)}_{m.group(2)}" if m else re.sub(r"[^A-Za-z0-9_.-]", "_", clone_url)


def _has_commit(repo: Path, sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}", cwd=repo, check=False).returncode == 0


def ensure_commits(clone_urls: list[str], shas: list[str]) -> Path:
    """Return a bare cache repo containing all `shas`, trying each remote in order."""
    if not clone_urls:
        raise CheckoutError("no clone URL")
    cache = CACHE_DIR / f"{_slug(clone_urls[0])}.git"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        _git("init", "--bare", "-q", str(cache))
        _git("config", "remote.origin.promisor", "true", cwd=cache)
        _git("config", "remote.origin.partialclonefilter", "blob:none", cwd=cache)
    missing = [s for s in shas if not _has_commit(cache, s)]
    errors = []
    for url in clone_urls:
        if not missing:
            break
        _git("config", "remote.origin.url", url, cwd=cache)
        proc = _git("fetch", "-q", "--depth=1", "--filter=blob:none", "origin", *missing, cwd=cache, check=False)
        if proc.returncode != 0:
            errors.append(f"{url}: {proc.stderr.strip()[:200]}")
        missing = [s for s in shas if not _has_commit(cache, s)]
    if missing:
        raise CheckoutError(f"commits {missing} not found in {clone_urls}: {'; '.join(errors)}")
    _git("config", "remote.origin.url", clone_urls[0], cwd=cache)
    return cache


def clone_urls(ex: PRExample) -> list[str]:
    urls = [ex.clone_url, f"https://github.com/{ex.repo}.git"]
    mirror = ex.metadata.get("mirror_url")
    if mirror:
        urls.insert(0, mirror)
    return list(dict.fromkeys(urls))


def git_diff(ex: PRExample) -> str:
    """base..head diff computed locally (fallback when the API refuses a very large diff).

    The cache is shallow, so this is a two-dot diff; both benchmarks give the PR's base commit.
    """
    cache = ensure_commits(clone_urls(ex), [ex.base_sha, ex.head_sha])
    return _git("diff", "--no-color", "--no-ext-diff", ex.base_sha, ex.head_sha, cwd=cache).stdout


def worktree_path(ex: PRExample, ref: str = "head") -> Path:
    safe = ex.id.split("/", 1)[-1]
    return REPOS_DIR / ex.benchmark / (safe if ref == "head" else f"{safe}@base")


def checkout(ex: PRExample, ref: str = "head", dest: Path | None = None, force: bool = False) -> Path:
    """Materialise the repo at ex.head_sha (or base_sha) and return its path."""
    if ref not in ("head", "base"):
        raise ValueError("ref must be 'head' or 'base'")
    sha = ex.head_sha if ref == "head" else ex.base_sha
    dest = dest or worktree_path(ex, ref)
    cache = ensure_commits(clone_urls(ex), [ex.base_sha, ex.head_sha])
    if dest.exists():
        current = _git("rev-parse", "HEAD", cwd=dest, check=False).stdout.strip()
        if current == sha and not force:
            return dest
        remove(dest, cache)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "-q", "--detach", "-f", str(dest), sha, cwd=cache)
    return dest


def remove(dest: Path, cache: Path | None = None) -> None:
    if cache is not None:
        _git("worktree", "remove", "--force", str(dest), cwd=cache, check=False)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    if cache is not None:
        _git("worktree", "prune", cwd=cache, check=False)
