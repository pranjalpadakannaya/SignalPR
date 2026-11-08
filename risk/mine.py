"""Mine AIDev-pop into a file-level dataset: which changed files draw human inline review comments.
(issue #10, tasks 2-4)

Cleaning
  * join the AIDev tables (pull_request, repository, pr_reviews, pr_review_comments, pr_files/
    pr_commit_details, related_issue, issue)
  * drop bot reviewers and bot comments (user_type != "User" or login ends in [bot]) and the PR
    author's own comments. Agent-authored PRs are kept: they are the subject of the study.
  * drop empty review comments (blank body)
  * drop oversized diffs: PRs with > 300 files, files with > 20,000 changed lines
  * drop unmerged PRs (unless --include-unmerged), open PRs, and PRs with no human review
Label: a file is 1 if a human (not the author, not a bot) left a non-empty inline comment on it.
Split: by repository (70/10/20, hashed) AND by time (train/val before the 80th-percentile
date, test = unseen repositories after it).

Output: data/processed/files.pkl (features + label + split), results/risk/dataset_stats.json
"""

from __future__ import annotations

import json
import time
import zlib

import numpy as np
import pandas as pd

from risk.features import FEATURES, PATCH_FEATURES, _nc, build_features, task_type_from_title
from risk.io import has_table, read_table
from risk.paths import INSIGHTS_PATH, PROCESSED_DIR

MAX_PR_FILES = 300  # oversized PRs (vendored code, mass renames) are dropped
MAX_FILE_CHANGES = 20_000

COMMENT_TOPICS = {
    "naming / readability": r"\b(name|naming|rename|readab|clarity|unclear|confusing)",
    "tests": r"\b(test|tests|coverage|assert|mock)",
    "error handling": r"\b(error|exception|throw|catch|handle|fail|null|none|undefined)",
    "docs / comments": r"\b(doc|docs|comment|docstring|readme|typo)",
    "performance": r"\b(perf|slow|fast|cache|alloc|memory|complexity|O\(n)",
    "security": r"\b(secur|auth|token|secret|inject|sanitiz|escape|permission|xss|csrf)",
    "simplify / remove": r"\b(simplif|unnecessary|not needed|remove|redundant|duplicate|over.?engineer"
    r"|dead code|unused)",
    "API / design": r"\b(api|interface|abstraction|design|pattern|signature|breaking|public)",
    "style / formatting": r"\b(style|format|indent|whitespace|lint|nit)",
    "logic / correctness": r"\b(bug|wrong|incorrect|should be|edge case|off.by|logic|race|instead)",
}


def _log(msg, t0):
    print(f"[mine {time.time() - t0:6.1f}s] {msg}", flush=True)


def _repo_bucket(repo_id) -> float:
    return (zlib.crc32(str(repo_id).encode()) % 10_000) / 10_000


def mine(include_unmerged: bool = False, time_quantile: float = 0.8, verbose: bool = True) -> dict:
    t0 = time.time()
    pr = read_table(
        "pull_request",
        [
            "id",
            "number",
            "title",
            "body_len",
            "body",
            "agent",
            "user",
            "state",
            "created_at",
            "closed_at",
            "merged_at",
            "repo_id",
            "repo_url",
        ],
    )
    if "body_len" not in pr:
        pr["body_len"] = pr.get("body", pd.Series("", index=pr.index)).fillna("").str.len()
    pr["body"] = pr.get("body", pd.Series("", index=pr.index)).fillna("").astype(str).str.slice(0, 400)
    repo = read_table("repository", ["id", "url", "full_name", "language", "stars", "forks"])
    reviews = read_table("pr_reviews", ["pr_id", "user", "user_type", "state"])
    rc = read_table("pr_review_comments", ["user", "user_type", "path", "pull_request_url", "body", "created_at"])
    _log(f"loaded {len(pr):,} PRs, {len(repo):,} repos, {len(reviews):,} reviews, {len(rc):,} inline comments", t0)

    # ---- link inline comments to PRs; keep human, non-author comments ----
    pr["pull_request_url"] = pr["repo_url"] + "/pulls/" + pr["number"].astype(str)
    rc = rc.merge(
        pr[["id", "pull_request_url", "user"]].rename(columns={"id": "pr_id", "user": "author"}),
        on="pull_request_url",
        how="inner",
    )
    is_bot = (rc["user_type"] != "User") | rc["user"].fillna("").str.endswith("[bot]")
    empty = rc["body"].fillna("").str.strip().eq("") if "body" in rc else pd.Series(False, index=rc.index)
    human_rc = rc[~is_bot & (rc["user"] != rc["author"]) & ~empty]
    _log(
        f"inline comments: {len(rc):,} linked, {int(is_bot.sum()):,} from bots, {int(empty.sum()):,} empty "
        f"-> {len(human_rc):,} human",
        t0,
    )
    bot_rc = rc[is_bot]

    rv = reviews.merge(pr[["id", "user"]].rename(columns={"id": "pr_id", "user": "author"}), on="pr_id")
    human_rv = rv[
        (rv["user_type"] == "User") & ~rv["user"].fillna("").str.endswith("[bot]") & (rv["user"] != rv["author"])
    ]
    reviewed = set(human_rv["pr_id"]) | set(human_rc["pr_id"])

    # ---- population ----
    pr["merged"] = pr["merged_at"].notna()
    pr["human_reviewed"] = pr["id"].isin(reviewed)
    pr = pr.merge(
        repo.rename(columns={"id": "repo_id", "language": "repo_language"})[
            ["repo_id", "repo_language", "stars", "forks", "full_name"]
        ],
        on="repo_id",
        how="left",
    )
    pr["issue_text"] = ""
    if has_table("related_issue"):
        ri = read_table("related_issue", ["pr_id", "issue_id"])
        pr["has_issue"] = pr["id"].isin(set(ri["pr_id"])).astype(int)
        if has_table("issue"):
            iss = read_table("issue", ["id", "title", "body"])
            iss["issue_text"] = iss["title"].fillna("") + " " + iss["body"].fillna("").str.slice(0, 1500)
            first = ri.drop_duplicates("pr_id").merge(iss[["id", "issue_text"]], left_on="issue_id", right_on="id")
            pr = pr.drop(columns="issue_text").merge(
                first[["pr_id", "issue_text"]].rename(columns={"pr_id": "id"}), on="id", how="left"
            )
            pr["issue_text"] = pr["issue_text"].fillna("")
    else:
        pr["has_issue"] = 0

    sel = pr["human_reviewed"] & (pr["state"] == "closed")
    if not include_unmerged:
        sel &= pr["merged"]
    pop = pr[sel].copy()
    _log(f"population: {len(pop):,} PRs (human-reviewed, closed{'' if include_unmerged else ', merged'})", t0)

    # ---- files ----
    files = read_table(
        "pr_files", ["pr_id", "filename", "status", "additions", "deletions", "changes", *PATCH_FEATURES]
    )
    has_patch = all(c in files.columns for c in PATCH_FEATURES)
    _log(
        "patch features: "
        + (
            "present (from download --with-patch)"
            if has_patch
            else "absent (light download) -> NaN, the model ignores them"
        ),
        t0,
    )
    n_all_files = len(files)
    files_all_pr = files  # kept for insights before filtering
    files = files[files["pr_id"].isin(set(pop["id"]))]
    nf = files.groupby("pr_id")["filename"].transform("size")
    files = files[(nf <= MAX_PR_FILES) & (files["changes"].fillna(0) <= MAX_FILE_CHANGES)].copy()
    _log(f"files: {len(files):,} rows for population (of {n_all_files:,})", t0)

    files = files.merge(
        pop[
            [
                "id",
                "agent",
                "title",
                "body",
                "body_len",
                "has_issue",
                "issue_text",
                "stars",
                "forks",
                "repo_language",
                "repo_id",
                "created_at",
            ]
        ].rename(columns={"id": "pr_id"}),
        on="pr_id",
        how="inner",
    )

    # ---- labels ----
    lab = human_rc.groupby(["pr_id", "path"]).size().rename("n_human_comments").reset_index()
    lab = lab.rename(columns={"path": "filename"})
    files = files.merge(lab, on=["pr_id", "filename"], how="left")
    files["n_human_comments"] = files["n_human_comments"].fillna(0).astype(np.int32)
    files["label"] = (files["n_human_comments"] > 0).astype(np.int8)
    blab = (
        bot_rc.groupby(["pr_id", "path"])
        .size()
        .rename("n_bot_comments")
        .reset_index()
        .rename(columns={"path": "filename"})
    )
    files = files.merge(blab, on=["pr_id", "filename"], how="left")
    files["n_bot_comments"] = files["n_bot_comments"].fillna(0).astype(np.int32)
    # Only PRs where a human left at least one inline comment carry file-level information
    # (a PR reviewed with only "LGTM" has all-negative files); we keep both but flag them.
    files["pr_has_inline"] = files.groupby("pr_id")["label"].transform("max").astype(np.int8)
    _log(f"labels: {files['label'].sum():,} commented files ({files['label'].mean():.2%})", t0)

    # ---- features ----
    feats = build_features(files)
    out = pd.concat(
        [
            feats,
            files[["label", "n_human_comments", "n_bot_comments", "pr_has_inline", "repo_id", "created_at"]]
            .reset_index(drop=True)
            .set_axis(feats.index),
        ],
        axis=1,
    )

    # ---- split: by repository AND by time ----
    rb = out["repo_id"].map(_repo_bucket)
    repo_split = np.where(rb < 0.7, "train", np.where(rb < 0.8, "val", "test"))
    cutoff = sorted(out["created_at"].dropna())[int(time_quantile * (out["created_at"].notna().sum() - 1))]
    early = out["created_at"] < cutoff
    split = np.full(len(out), "unused", dtype=object)
    split[(repo_split == "train") & early] = "train"
    split[(repo_split == "val") & early] = "val"
    split[(repo_split == "test") & ~early] = "test"
    out["split"] = split
    out["repo_split"] = repo_split
    _log(
        "split sizes: "
        + ", ".join(f"{k}={v:,}" for k, v in out["split"].value_counts().items())
        + f" | time cutoff {cutoff}",
        t0,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_pickle(PROCESSED_DIR / "files.pkl")
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    insights = build_insights(pr, pop, files_all_pr, out, human_rc, bot_rc, human_rv, cutoff, include_unmerged)
    insights["patch_features"] = has_patch
    INSIGHTS_PATH.write_text(json.dumps(insights, indent=1, default=float))
    _log(f"wrote {PROCESSED_DIR / 'files.pkl'} and {INSIGHTS_PATH}", t0)
    return insights


def _rate_table(df, key, label="label", min_n=200, top=None):
    g = df.groupby(key)[label].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n].sort_values("mean", ascending=False)
    if top:
        g = g.head(top)
    return [{"key": str(r[key]), "rate": float(r["mean"]), "n": int(r["size"])} for _, r in g.iterrows()]


def build_insights(pr, pop, files_all, out, human_rc, bot_rc, human_rv, cutoff, include_unmerged):
    agents = []
    rc_per_pr = human_rc.groupby("pr_id").size()
    for a, sub in pr.groupby("agent"):
        ids = sub["id"]
        agents.append(
            {
                "agent": a,
                "prs": int(len(sub)),
                "merge_rate": float(sub["merged"].mean()),
                "closed_unmerged_rate": float(((sub["state"] == "closed") & ~sub["merged"]).mean()),
                "human_review_rate": float(sub["human_reviewed"].mean()),
                "inline_comment_pr_rate": float(ids.isin(rc_per_pr.index).mean()),
                "comments_per_reviewed_pr": float(rc_per_pr.reindex(ids[sub["human_reviewed"].values]).fillna(0).mean())
                if sub["human_reviewed"].any()
                else 0.0,
                "repos": int(sub["repo_id"].nunique()),
            }
        )
    agents.sort(key=lambda r: -r["prs"])

    month = pr["created_at"].str.slice(0, 7)
    monthly = pr.assign(month=month).groupby(["month", "agent"]).size().unstack(fill_value=0)
    monthly_rows = [{"month": m, **{a: int(v) for a, v in row.items()}} for m, row in monthly.iterrows()]

    lab = out[out["pr_has_inline"] == 1]  # files of PRs that received inline comments
    cat = np.select(
        [
            lab["is_generated"] == 1,
            lab["is_lock"] == 1,
            lab["is_test"] == 1,
            lab["is_doc"] == 1,
            lab["is_ci"] == 1,
            lab["is_build"] == 1,
            lab["is_config"] == 1,
            lab["is_asset"] == 1,
            lab["is_code"] == 1,
        ],
        ["generated/vendored", "lockfile", "test", "docs", "CI", "build", "config", "asset", "source code"],
        "other",
    )
    bins = [-1, 5, 20, 50, 100, 250, 500, 1e12]
    names = ["1-5", "6-20", "21-50", "51-100", "101-250", "251-500", "500+"]
    churn_b = pd.cut(lab["changes"], bins=bins, labels=names)
    nb = pd.cut(
        out["pr_n_files"],
        bins=[0, 1, 3, 5, 10, 20, 50, 1e9],
        labels=["1", "2-3", "4-5", "6-10", "11-20", "21-50", "50+"],
    )

    topics = []
    bodies = human_rc["body"].fillna("").str.lower()
    for name, pat in COMMENT_TOPICS.items():
        topics.append({"topic": name, "share": float(bodies.str.contains(_nc(pat), regex=True).mean())})
    topics.sort(key=lambda r: -r["share"])

    return {
        "source": "AIDev-pop (hao-li/AIDev v5, repositories with >100 stars)",
        "synthetic": False,
        "totals": {
            "prs": int(len(pr)),
            "repos": int(pr["repo_id"].nunique()),
            "agents": int(pr["agent"].nunique()),
            "file_changes": int(len(files_all)),
            "inline_comments_human": int(len(human_rc)),
            "inline_comments_bot": int(len(bot_rc)),
            "human_reviews": int(len(human_rv)),
            "population_prs": int(len(pop)),
            "population_files": int(len(out)),
            "positive_files": int(out["label"].sum()),
            "positive_rate": float(out["label"].mean()),
            "date_min": str(pr["created_at"].min()),
            "date_max": str(pr["created_at"].max()),
            "time_cutoff": str(cutoff),
            "include_unmerged": include_unmerged,
        },
        "splits": {
            k: {
                "files": int((out["split"] == k).sum()),
                "prs": int(out.loc[out["split"] == k, "pr_id"].nunique()),
                "repos": int(out.loc[out["split"] == k, "repo_id"].nunique()),
                "positive_rate": float(out.loc[out["split"] == k, "label"].mean() if (out["split"] == k).any() else 0),
            }
            for k in ["train", "val", "test"]
        },
        "agents": agents,
        "monthly": monthly_rows,
        "rate_by_category": _rate_table(lab.assign(cat=cat), "cat", min_n=50),
        "rate_by_lang": _rate_table(lab, "lang", min_n=300, top=16),
        "rate_by_churn": [
            {"key": k, "rate": float(v["mean"]), "n": int(v["size"])}
            for k, v in lab.groupby(churn_b, observed=False)["label"].agg(["mean", "size"]).iterrows()
        ],
        "rate_by_pr_size": [
            {"key": k, "rate": float(v["mean"]), "n": int(v["size"])}
            for k, v in out.groupby(nb, observed=False)["label"].agg(["mean", "size"]).iterrows()
        ],
        "rate_by_status": _rate_table(lab, "status", min_n=50),
        "rate_by_agent": _rate_table(lab, "agent", min_n=50),
        "comment_topics": topics,
        "task_types": pr.assign(t=task_type_from_title(pr["title"]))["t"].value_counts().head(10).to_dict(),
        "features": FEATURES,
    }


if __name__ == "__main__":
    mine()
