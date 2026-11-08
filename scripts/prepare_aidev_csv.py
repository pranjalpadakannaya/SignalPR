"""
Download the AIDev-pop tables SignalPR needs and save them as small .csv.gz files.

Run on your laptop / Kaggle / ASU Sol (needs internet):
    pip install -U pandas pyarrow huggingface_hub
    python prepare_aidev_upload.py

Output: folder ./aidev_upload/ (~150-400 MB). Copy its *.csv.gz files into data/raw/, then run
`python -m risk mine`. (Alternative with pyarrow installed: `python -m risk download`.)
Download traffic is ~2-3 GB: the 12.4 GB pr_commit_details table is streamed with only
its light columns (the huge `patch` column is never fetched).
"""

import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO, REV = "hao-li/AIDev", "v5"
OUT = Path("aidev_upload")
OUT.mkdir(exist_ok=True)


def local(name):
    print(f"  downloading {name}.parquet ...", flush=True)
    return hf_hub_download(REPO, f"{name}.parquet", repo_type="dataset", revision=REV)


def cols_present(path_or_file, wanted):
    names = set(pq.ParquetFile(path_or_file).schema_arrow.names)
    missing = [c for c in wanted if c not in names]
    if missing:
        print(f"    (columns not in this release, skipped: {missing})")
    return [c for c in wanted if c in names]


def save(df, name):
    p = OUT / f"{name}.csv.gz"
    df.to_csv(p, index=False, compression="gzip")
    print(f"  -> {p}  {len(df):,} rows  {p.stat().st_size / 1e6:.1f} MB", flush=True)


t0 = time.time()

# 1) pull requests of popular repos (>100 stars); body reduced to its length + first 400 chars
print("[1/6] pull_request")
p = local("pull_request")
c = cols_present(
    p,
    [
        "id",
        "number",
        "title",
        "body",
        "agent",
        "user",
        "state",
        "created_at",
        "closed_at",
        "merged_at",
        "repo_id",
        "repo_url",
        "html_url",
    ],
)
pr = pq.read_table(p, columns=c).to_pandas()
if "body" in pr:
    pr["body_len"] = pr["body"].fillna("").str.len()
    pr["body"] = pr["body"].fillna("").str.slice(0, 400)
save(pr, "pull_request")
pr_ids = set(pr["id"])

# 2) repositories
print("[2/6] repository")
p = local("repository")
save(pq.read_table(p).to_pandas(), "repository")

# 3) reviews (no body)
print("[3/6] pr_reviews")
p = local("pr_reviews")
c = cols_present(p, ["id", "pr_id", "user", "user_type", "state", "submitted_at"])
save(pq.read_table(p, columns=c).to_pandas(), "pr_reviews")

# 4) inline review comments (the labels); body trimmed, diff_hunk dropped
print("[4/6] pr_review_comments")
p = local("pr_review_comments")
c = cols_present(
    p,
    [
        "id",
        "pull_request_review_id",
        "user",
        "user_type",
        "path",
        "line",
        "original_line",
        "position",
        "pull_request_url",
        "created_at",
        "body",
    ],
)
rc = pq.read_table(p, columns=c).to_pandas()
if "body" in rc:
    rc["body"] = rc["body"].fillna("").str.slice(0, 300)
save(rc, "pr_review_comments")

# 5) linked issues (for the intent checker)
print("[5/6] related_issue + issue")
try:
    save(pq.read_table(local("related_issue")).to_pandas(), "related_issue")
    p = local("issue")
    c = cols_present(p, ["id", "number", "title", "body", "state", "created_at"])
    iss = pq.read_table(p, columns=c).to_pandas()
    if "body" in iss:
        iss["body"] = iss["body"].fillna("").str.slice(0, 1500)
    save(iss, "issue")
except Exception as e:  # optional tables
    print("    skipped:", e)

# 6) per-file changes, streamed from the 12.4 GB table WITHOUT the patch column,
#    aggregated to one row per (pr_id, filename)
print("[6/6] pr_commit_details (streaming light columns, takes a while)")
fs = HfFileSystem()
f = fs.open(f"datasets/{REPO}@{REV}/pr_commit_details.parquet", "rb", block_size=16 * 2**20)
pf = pq.ParquetFile(f)
c = cols_present(f, ["pr_id", "sha", "filename", "status", "additions", "deletions", "changes"])
parts, seen = [], 0
for i, batch in enumerate(pf.iter_batches(batch_size=262_144, columns=c)):
    df = batch.to_pandas()
    seen += len(df)
    df = df[df["pr_id"].isin(pr_ids) & df["filename"].notna()]
    if df.empty:
        continue
    agg = (
        df.groupby(["pr_id", "filename"], sort=False)
        .agg(
            status=("status", "first"),
            additions=("additions", "sum"),
            deletions=("deletions", "sum"),
            changes=("changes", "sum"),
            n_commits=("sha", "nunique"),
        )
        .reset_index()
    )
    parts.append(agg)
    if i % 10 == 0:
        print(f"    {seen:,} rows scanned, {time.time() - t0:.0f}s", flush=True)
files = pd.concat(parts, ignore_index=True)
files = (
    files.groupby(["pr_id", "filename"], sort=False)
    .agg(
        status=("status", "first"),
        additions=("additions", "sum"),
        deletions=("deletions", "sum"),
        changes=("changes", "sum"),
        n_commits=("n_commits", "sum"),
    )
    .reset_index()
)
save(files, "pr_files")

total = sum(x.stat().st_size for x in OUT.glob("*.csv.gz")) / 1e6
print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Folder {OUT}/ = {total:.0f} MB. Zip it and upload it.")
