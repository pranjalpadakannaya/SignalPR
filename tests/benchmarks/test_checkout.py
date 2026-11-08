"""Checkout against a local git repo (no network)."""

import shutil
import subprocess

import pytest

from benchmarks import checkout as co
from tests.benchmarks.test_schema_registry import make_example

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    git("config", "user.email", "t@example.com", cwd=repo)
    git("config", "user.name", "t", cwd=repo)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=repo)
    git("config", "uploadpack.allowFilter", "true", cwd=repo)
    (repo / "a.py").write_text("x = 1\n")
    git("add", ".", cwd=repo)
    git("commit", "-q", "-m", "base", cwd=repo)
    base = git("rev-parse", "HEAD", cwd=repo)
    (repo / "a.py").write_text("x = 2\n")
    (repo / "b.py").write_text("y = 1\n")
    git("add", ".", cwd=repo)
    git("commit", "-q", "-m", "head", cwd=repo)
    head = git("rev-parse", "HEAD", cwd=repo)
    return repo, base, head


@pytest.fixture
def repos_dir(tmp_path, monkeypatch):
    d = tmp_path / "repos"
    monkeypatch.setattr(co, "REPOS_DIR", d)
    monkeypatch.setattr(co, "CACHE_DIR", d / ".cache")
    return d


def test_checkout_head_and_base(upstream, repos_dir):
    repo, base, head = upstream
    ex = make_example(clone_url=f"file://{repo}", base_sha=base, head_sha=head, repo="local/upstream")
    path = co.checkout(ex)
    assert path == repos_dir / "martian" / "o_r_1"
    assert (path / "b.py").exists() and (path / "a.py").read_text() == "x = 2\n"
    assert co.checkout(ex) == path  # idempotent
    base_path = co.checkout(ex, ref="base")
    assert not (base_path / "b.py").exists() and (base_path / "a.py").read_text() == "x = 1\n"


def test_git_diff(upstream, repos_dir):
    repo, base, head = upstream
    ex = make_example(clone_url=f"file://{repo}", base_sha=base, head_sha=head, repo="local/upstream")
    diff = co.git_diff(ex)
    assert "+x = 2" in diff and "b/b.py" in diff


def test_missing_commit_raises(upstream, repos_dir):
    repo, base, _ = upstream
    ex = make_example(clone_url=f"file://{repo}", base_sha=base, head_sha="f" * 40, repo="local/upstream")
    with pytest.raises(co.CheckoutError):
        co.checkout(ex)
