"""File-level features for the risk model.

ONE implementation used in two places:
  * mining AIDev-pop (training)       -> build_features(files_df)
  * reviewing a live pull request     -> build_features(files_df_from_github_or_diff)

Issue #10 feature groups: lines +/-, file counts (PR context), complexity change, repeated code,
tool warnings, issue alignment, agent name, plus path semantics and repository metadata.

Only information available both in AIDev and in a live PR *before review* is used:
path, change status, lines added/removed, the PR's other files, PR metadata (agent,
title, description length, linked issue) and repository metadata (stars, forks, language).
Per-file commit counts are deliberately NOT used: in AIDev they include commits pushed
after the review (fixes in response to comments), which would leak the label.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from tools.lang import CODE_LANGS, EXT_LANG, language_of  # noqa: F401  (re-exported)
from tools.rules import run_rules

REPO_LANG_NORM = {
    "Python": "python",
    "Jupyter Notebook": "python",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "Java": "java",
    "Kotlin": "kotlin",
    "Go": "go",
    "Rust": "rust",
    "C": "c",
    "C++": "cpp",
    "C#": "csharp",
    "Swift": "swift",
    "Ruby": "ruby",
    "PHP": "php",
    "Dart": "dart",
    "Shell": "shell",
    "Scala": "scala",
    "Lua": "lua",
    "Vue": "vue",
    "Svelte": "svelte",
    "HTML": "html",
    "CSS": "css",
    "Elixir": "elixir",
    "Haskell": "haskell",
    "Objective-C": "objc",
    "F#": "fsharp",
    "Zig": "zig",
    "R": "r",
    "Julia": "julia",
    "Solidity": "solidity",
    "PowerShell": "powershell",
    "MDX": "markdown",
}


# Directory / name tokens that carry signal about how reviewers treat a file
def _nc(pattern: str) -> str:
    """Make every group non-capturing (pandas warns on capture groups in str.contains)."""
    return re.sub(r"(?<!\\)\((?!\?)", "(?:", pattern)


PATH_TOKENS = {
    "tok_src": r"(^|/)(src|lib|pkg|internal|app|core)(/|$)",
    "tok_api": r"(^|/)(api|apis|routes?|controllers?|handlers?|endpoints?|server)(/|$|\.)",
    "tok_ui": r"(^|/)(components?|ui|views?|pages?|screens?|widgets?|frontend|web)(/|$)",
    "tok_util": r"(^|/)(utils?|helpers?|common|shared|misc)(/|$|\.)",
    "tok_model": r"(^|/)(models?|schemas?|entities|types?|dto)(/|$|\.)",
    "tok_db": r"(^|/)(db|database|migrations?|sql|repositor(y|ies)|store)(/|$|\.)",
    "tok_auth": r"(auth|login|session|token|password|crypt|secur|permission|oauth|jwt|acl)",
    "tok_config": r"(^|/)(config|configs|settings|conf)(/|$|\.)",
    "tok_example": r"(^|/)(examples?|samples?|demo|playground|benchmarks?)(/|$)",
    "tok_scripts": r"(^|/)(scripts?|tools?|bin|hack)(/|$)",
    "tok_public_api": r"(^|/)(public|include|exports?|index\.(ts|js)|__init__\.py|mod\.rs)",
}
_TOK_RE = {k: re.compile(_nc(v), re.I) for k, v in PATH_TOKENS.items()}

TEST_RE = re.compile(
    _nc(
        r"(^|/)(tests?|__tests__|spec|specs|testing|e2e|integration-tests?)(/|$)|(_test|\.test|\.spec|_spec|Tests?)\.\w+$|(^|/)test_[^/]+$"
    ),
    re.I,
)
DOC_RE = re.compile(
    _nc(
        r"(^|/)(docs?|documentation|wiki|guides?)(/|$)|(^|/)(README|CHANGELOG|CHANGES|CONTRIBUTING|LICENSE|NOTICE|AUTHORS|HISTORY)[^/]*$"
    ),
    re.I,
)
CI_RE = re.compile(
    _nc(
        r"(^|/)(\.github|\.gitlab|\.circleci|\.buildkite|\.azure-pipelines)(/|$)|(^|/)(Jenkinsfile|\.travis\.yml|azure-pipelines\.yml)$"
    ),
    re.I,
)
LOCK_RE = re.compile(
    _nc(
        r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|Gemfile\.lock|composer\.lock|go\.sum|uv\.lock|Pipfile\.lock|\.lock)$"
    ),
    re.I,
)
GEN_RE = re.compile(
    _nc(
        r"(\.min\.(js|css)$|(^|/)(dist|build|out|vendor|node_modules|generated|gen|__generated__|third_party)(/|$)|\.g\.dart$|_pb2\.py$|\.pb\.go$|\.designer\.cs$|\.snap$)"
    ),
    re.I,
)
BUILD_RE = re.compile(
    _nc(
        r"(^|/)(Dockerfile|Makefile|CMakeLists\.txt|package\.json|pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.txt|build\.gradle(\.kts)?|pom\.xml|Cargo\.toml|go\.mod|tsconfig[^/]*\.json|\.csproj)$"
    ),
    re.I,
)
AGENT_RULES_RE = re.compile(
    _nc(r"(^|/)(AGENTS\.md|CLAUDE\.md|\.cursorrules|\.cursor/|copilot-instructions\.md|\.windsurfrules|GEMINI\.md)"),
    re.I,
)

TITLE_TYPE_RE = re.compile(r"^\s*\[?(feat|fix|docs|refactor|test|tests|chore|perf|build|ci|style|revert)\b", re.I)
TITLE_KEYWORDS = {
    "fix": r"\b(fix|bug|error|issue|crash|broken|resolve|patch)",
    "feat": r"\b(add|implement|support|introduce|create|new|feature|enable)",
    "docs": r"\b(doc|docs|readme|documentation|comment|typo)",
    "refactor": r"\b(refactor|clean|cleanup|restructure|simplif|rename|move|extract)",
    "test": r"\b(test|tests|coverage|spec)",
    "chore": r"\b(bump|upgrade|update dep|dependenc|chore|version|release)",
    "perf": r"\b(perf|performance|optimi[sz]|speed|faster|cache)",
}

CATEGORICAL = ["agent", "lang", "status", "repo_lang", "task_type"]
PATCH_FEATURES = [
    "complexity_delta",
    "branches_added",
    "repeated_added_lines",
    "repeated_ratio",
    "tool_warnings",
    "tool_security",
    "tool_critical",
    "n_hunks",
]
NUMERIC = [
    "additions",
    "deletions",
    "changes",
    "log_changes",
    "add_ratio",
    "is_new",
    "is_removed",
    "is_renamed",
    "depth",
    "name_len",
    "is_test",
    "is_doc",
    "is_ci",
    "is_lock",
    "is_generated",
    "is_build",
    "is_config",
    "is_code",
    "is_asset",
    "is_agent_rules",
    *PATH_TOKENS.keys(),
    "pr_n_files",
    "pr_additions",
    "pr_deletions",
    "pr_log_changes",
    "churn_share",
    "churn_rank_pct",
    "is_largest_file",
    "pr_n_code_files",
    "pr_test_share",
    "pr_has_tests",
    "pr_n_dirs",
    "pr_n_langs",
    "same_dir_files",
    "lang_matches_repo",
    "title_len",
    "body_len_log",
    "has_issue",
    "repo_stars_log",
    "repo_forks_log",
    # issue alignment (computed from the linked issue, else the PR title/description)
    "issue_path_overlap",
    "issue_path_hits",
    "pr_issue_overlap",
    # patch-content features (NaN when the patch text is not available, e.g. light AIDev download)
    *PATCH_FEATURES,
]
FEATURES = NUMERIC + CATEGORICAL


def _ext(path: pd.Series) -> pd.Series:
    base = path.str.rsplit("/", n=1).str[-1].str.lower()
    ext = base.str.extract(r"\.([a-z0-9_+-]{1,12})$", expand=False)
    special = base.map({"dockerfile": "dockerfile", "makefile": "mk", "jenkinsfile": "build"})
    return ext.fillna(special).fillna("")


def task_type_from_title(title: pd.Series) -> pd.Series:
    t = title.fillna("").astype(str)
    out = t.str.extract(TITLE_TYPE_RE, expand=False).str.lower().replace({"tests": "test"})
    for name, pat in TITLE_KEYWORDS.items():
        out = out.where(out.notna(), np.where(t.str.contains(_nc(pat), case=False, regex=True), name, None))
    return out.fillna("other")


def _col(df: pd.DataFrame, name: str, default) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df has one row per (pr_id, filename) with columns:
    pr_id, filename, status, additions, deletions, [changes],
    and optional PR/repo columns: agent, title, body_len, has_issue, stars, forks, repo_language.
    Returns a DataFrame with FEATURES (+ pr_id, filename passthrough)."""
    d = pd.DataFrame(index=df.index)
    path = df["filename"].astype(str)
    d["pr_id"] = df["pr_id"].values
    d["filename"] = path.values

    add = pd.to_numeric(df["additions"], errors="coerce").fillna(0).clip(lower=0)
    dele = pd.to_numeric(df["deletions"], errors="coerce").fillna(0).clip(lower=0)
    chg = pd.to_numeric(df["changes"], errors="coerce").fillna(add + dele) if "changes" in df.columns else add + dele
    d["additions"], d["deletions"], d["changes"] = add, dele, chg
    d["log_changes"] = np.log1p(chg)
    d["add_ratio"] = (add / (chg.replace(0, np.nan))).fillna(0)

    status = _col(df, "status", "modified").fillna("modified").astype(str).str.lower()
    d["status"] = status
    d["is_new"] = (status == "added").astype(np.int8)
    d["is_removed"] = (status == "removed").astype(np.int8)
    d["is_renamed"] = (status == "renamed").astype(np.int8)

    ext = _ext(path)
    lang = ext.map(EXT_LANG).fillna(pd.Series(np.where(ext == "", "none", "other"), index=ext.index))
    d["lang"] = lang
    d["depth"] = path.str.count("/")
    d["name_len"] = path.str.rsplit("/", n=1).str[-1].str.len()
    d["is_test"] = path.str.contains(TEST_RE).astype(np.int8)
    d["is_doc"] = (path.str.contains(DOC_RE) | lang.isin(["markdown", "docs"])).astype(np.int8)
    d["is_ci"] = path.str.contains(CI_RE).astype(np.int8)
    d["is_lock"] = path.str.contains(LOCK_RE).astype(np.int8)
    d["is_generated"] = path.str.contains(GEN_RE).astype(np.int8)
    d["is_build"] = path.str.contains(BUILD_RE).astype(np.int8)
    d["is_config"] = (lang == "config").astype(np.int8)
    d["is_code"] = lang.isin(CODE_LANGS).astype(np.int8)
    d["is_asset"] = (lang == "asset").astype(np.int8)
    d["is_agent_rules"] = path.str.contains(AGENT_RULES_RE).astype(np.int8)
    for k, rx in _TOK_RE.items():
        d[k] = path.str.contains(rx).astype(np.int8)

    # ---- PR context (computed over the PR's files) ----
    g = d.groupby("pr_id", sort=False)
    d["pr_n_files"] = g["filename"].transform("size")
    d["pr_additions"] = g["additions"].transform("sum")
    d["pr_deletions"] = g["deletions"].transform("sum")
    pr_changes = g["changes"].transform("sum")
    d["pr_log_changes"] = np.log1p(pr_changes)
    d["churn_share"] = (d["changes"] / pr_changes.replace(0, np.nan)).fillna(0)
    d["churn_rank_pct"] = g["changes"].rank(pct=True, method="average")
    d["is_largest_file"] = (g["changes"].transform("max") == d["changes"]).astype(np.int8)
    d["pr_n_code_files"] = g["is_code"].transform("sum")
    d["pr_test_share"] = g["is_test"].transform("mean")
    d["pr_has_tests"] = (g["is_test"].transform("max") > 0).astype(np.int8)
    dirs = path.str.rsplit("/", n=1).str[0].where(path.str.contains("/"), "")
    d["_dir"] = dirs.values
    d["pr_n_dirs"] = d.groupby("pr_id", sort=False)["_dir"].transform("nunique")
    d["pr_n_langs"] = d.assign(_l=lang.values).groupby("pr_id", sort=False)["_l"].transform("nunique")
    d["same_dir_files"] = d.groupby(["pr_id", "_dir"], sort=False)["filename"].transform("size") - 1
    d = d.drop(columns="_dir")

    # ---- PR & repo metadata ----
    d["agent"] = _col(df, "agent", "unknown").fillna("unknown").astype(str)
    title = _col(df, "title", "")
    d["title_len"] = title.fillna("").astype(str).str.len()
    d["task_type"] = task_type_from_title(title)
    d["body_len_log"] = np.log1p(pd.to_numeric(_col(df, "body_len", 0), errors="coerce").fillna(0))
    d["has_issue"] = pd.to_numeric(_col(df, "has_issue", 0), errors="coerce").fillna(0).astype(np.int8)
    d["repo_stars_log"] = np.log1p(pd.to_numeric(_col(df, "stars", 0), errors="coerce").fillna(0))
    d["repo_forks_log"] = np.log1p(pd.to_numeric(_col(df, "forks", 0), errors="coerce").fillna(0))
    rl = _col(df, "repo_language", None).map(REPO_LANG_NORM)
    d["repo_lang"] = rl.fillna("other")
    d["lang_matches_repo"] = (rl.values == lang.values).astype(np.int8)

    # ---- issue alignment (token sets computed once per PR, not per file: memory-safe on 1.5M rows) ----
    issue_text = _col(df, "issue_text", "").fillna("").astype(str).values
    pr_text = (title.fillna("").astype(str) + " " + _col(df, "body", "").fillna("").astype(str)).values
    titles = title.fillna("").astype(str).values
    ref_tok: dict = {}
    title_overlap: dict = {}
    hits = np.zeros(len(df), dtype=np.float32)
    plen = np.zeros(len(df), dtype=np.float32)
    pri = np.full(len(df), np.nan, dtype=np.float32)
    for i, (pid, pth) in enumerate(zip(d["pr_id"].values, path.values)):
        if pid not in ref_tok:
            has = len(issue_text[i]) > 0
            ref_tok[pid] = _tokens(issue_text[i] if has else pr_text[i])
            tt = _tokens(titles[i])
            title_overlap[pid] = (len(tt & ref_tok[pid]) / max(1, len(tt))) if has else np.nan
        ptok = _tokens(pth)
        hits[i] = len(ptok & ref_tok[pid])
        plen[i] = len(ptok)
        pri[i] = title_overlap[pid]
    d["issue_path_hits"] = hits
    d["issue_path_overlap"] = hits / np.maximum(1, plen)
    d["pr_issue_overlap"] = pri
    del ref_tok, title_overlap

    # ---- patch-content features: from precomputed columns, or computed here from `patch` ----
    if "patch" in df.columns:
        pf = pd.DataFrame([patch_features(p, f) for p, f in zip(df["patch"].values, path.values)], index=df.index)
        for c in PATCH_FEATURES:
            d[c] = pf[c].values
    else:
        for c in PATCH_FEATURES:
            d[c] = pd.to_numeric(_col(df, c, np.nan), errors="coerce").values

    for c in NUMERIC:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return d


# ------------------------------------------------------------------ helpers for the new features
_STOP = set(
    """the and for with that this from into when then else none null true false self return import
def class function const let var new use add fix update file files code src lib main index test tests""".split()
)


def _tokens(text: str) -> set:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    out = set()
    for t in re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text.replace("_", " ").replace("-", " ").replace("/", " ")):
        t = t.lower()
        if t not in _STOP:
            out.add(t[:-1] if t.endswith("s") and len(t) > 4 else t)
    return out


_BRANCH = re.compile(r"\b(if|elif|else if|for|foreach|while|case|catch|except|when)\b|&&|\|\||\?\s|\band\b|\bor\b")


def patch_features(patch, path: str) -> dict:
    """Complexity change, repeated code and tool warnings from one file's unified-diff patch."""
    if not isinstance(patch, str) or not patch:
        return {c: np.nan for c in PATCH_FEATURES}
    added, removed, hunks, new_no = [], [], 0, 0
    for line in patch.splitlines():
        if line.startswith("@@"):
            hunks += 1
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
            new_no = int(m.group(1)) if m else 0
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append((new_no, line[1:]))
            new_no += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif not line.startswith("\\"):
            new_no += 1
    b_add = sum(len(_BRANCH.findall(t)) for _, t in added)
    b_rem = sum(len(_BRANCH.findall(t)) for t in removed)
    norm = [re.sub(r"\s+", " ", t.strip()) for _, t in added]
    norm = [t for t in norm if len(t) > 12 and not t.startswith(("#", "//", "import ", "from ", "*"))]
    counts: dict[str, int] = {}
    for t in norm:
        counts[t] = counts.get(t, 0) + 1
    repeated = sum(c for c in counts.values() if c > 1)
    hits = list(run_rules(language_of(path), added))
    return {
        "complexity_delta": float(b_add - b_rem),
        "branches_added": float(b_add),
        "repeated_added_lines": float(repeated),
        "repeated_ratio": repeated / max(1, len(norm)),
        "tool_warnings": float(len(hits)),
        "tool_security": float(sum(r.category == "security" for r, _, _ in hits)),
        "tool_critical": float(sum(r.severity == "critical" for r, _, _ in hits)),
        "n_hunks": float(hunks),
    }
