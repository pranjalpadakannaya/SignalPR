"""Download AIDev-pop into data/raw as parquet, keeping only the columns SignalPR uses.

    python -m risk download                 # light: ~2-3 GB of traffic, no patch text
    python -m risk download --with-patch    # streams the 12 GB patch column once and keeps
                                                     # only the patch features (complexity change,
                                                     # repeated code, tool warnings)

pr_commit_details is read over HTTP range requests with column projection and aggregated to one
row per (pr_id, filename) as data/raw/pr_files.parquet. Needs: pip install pyarrow huggingface_hub
"""

from __future__ import annotations

import time

from risk.features import PATCH_FEATURES, patch_features
from risk.paths import HF_REPO, HF_REVISION, RAW_DIR

TABLES = {
    "pull_request": [
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
    "repository": None,
    "pr_reviews": ["id", "pr_id", "user", "user_type", "state", "submitted_at"],
    "pr_review_comments": [
        "id",
        "pull_request_review_id",
        "user",
        "user_type",
        "path",
        "line",
        "position",
        "pull_request_url",
        "created_at",
        "body",
    ],
    "related_issue": None,
    "issue": ["id", "number", "title", "body", "state", "created_at"],
}


def download(revision: str | None = None, with_patch: bool = False):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    rev = revision or HF_REVISION
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, cols in TABLES.items():
        print(f"[download] {name}", flush=True)
        p = hf_hub_download(HF_REPO, f"{name}.parquet", repo_type="dataset", revision=rev)
        names = set(pq.ParquetFile(p).schema_arrow.names)
        tbl = pq.read_table(p, columns=[c for c in cols if c in names] if cols else None)
        if name == "pull_request" and "body" in tbl.column_names:
            df = tbl.to_pandas()
            df["body_len"] = df["body"].fillna("").str.len()
            df["body"] = df["body"].fillna("").str.slice(0, 400)
            tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl, RAW_DIR / f"{name}.parquet")
    pr_ids = set(pq.read_table(RAW_DIR / "pull_request.parquet", columns=["id"]).column("id").to_pylist())

    print("[download] pr_commit_details (streaming light columns)", flush=True)
    f = HfFileSystem().open(f"datasets/{HF_REPO}@{rev}/pr_commit_details.parquet", "rb", block_size=16 * 2**20)
    pf = pq.ParquetFile(f)
    want = ["pr_id", "sha", "filename", "status", "additions", "deletions", "changes"] + (
        ["patch"] if with_patch else []
    )
    cols = [c for c in want if c in pf.schema_arrow.names]
    parts, seen = [], 0
    for i, batch in enumerate(pf.iter_batches(batch_size=262_144, columns=cols)):
        df = batch.to_pandas()
        seen += len(df)
        df = df[df["pr_id"].isin(pr_ids) & df["filename"].notna()]
        if len(df):
            agg = dict(
                status=("status", "first"),
                additions=("additions", "sum"),
                deletions=("deletions", "sum"),
                changes=("changes", "sum"),
                n_commits=("sha", "nunique"),
            )
            if "patch" in df:
                pf_rows = pd.DataFrame(
                    [patch_features(p, f) for p, f in zip(df["patch"].values, df["filename"].values)], index=df.index
                )
                df = pd.concat([df.drop(columns="patch"), pf_rows], axis=1)
                agg.update({c: (c, "sum") for c in PATCH_FEATURES if c != "repeated_ratio"})
                agg["repeated_ratio"] = ("repeated_ratio", "max")
            parts.append(df.groupby(["pr_id", "filename"], sort=False).agg(**agg).reset_index())
        if i % 10 == 0:
            print(f"  {seen:,} rows scanned ({time.time() - t0:.0f}s)", flush=True)
    files = pd.concat(parts, ignore_index=True)
    agg = dict(
        status=("status", "first"),
        additions=("additions", "sum"),
        deletions=("deletions", "sum"),
        changes=("changes", "sum"),
        n_commits=("n_commits", "sum"),
    )
    if with_patch:
        agg.update({c: (c, "sum") for c in PATCH_FEATURES if c != "repeated_ratio"})
        agg["repeated_ratio"] = ("repeated_ratio", "max")
    files = files.groupby(["pr_id", "filename"], sort=False).agg(**agg).reset_index()
    files.to_parquet(RAW_DIR / "pr_files.parquet", index=False)
    print(f"[download] done in {(time.time() - t0) / 60:.1f} min → {RAW_DIR}")
