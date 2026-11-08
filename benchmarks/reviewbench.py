"""GitHub ReviewBench (full set, 219 PRs) -> PRExample. TEST SET ONLY (issue #5).

Never tune prompts, thresholds or anything else on these PRs: load them only through
`load_benchmark("reviewbench", purpose="eval")` for the final frozen configuration.

Source: github.com/review-bench/ReviewBench, pinned in benchmarks.yaml.
- corpus/manifest.json: the 219 PRs (repo, number, base/head SHA, title, body, language, size)
- golden/<pr_key>.json: findings with file + line range on head, severity (high/medium/low),
  category, tp_fp label, scope, difficulty, context needed and the producer that found it.
  Judged false positives are kept (label="fp") because grounded precision penalises matching them.
- corpus/test/test.json: the 25-PR subset ReviewBench uses for adapter smoke tests.

Commits are fetched from the review-bench/<owner>_<repo> mirrors (the upstream repo is the
fallback). PR author / linked issues come from the upstream PR when it still exists.

    python -m benchmarks download reviewbench
"""

from __future__ import annotations

import io
import json
import tarfile

from benchmarks import github
from benchmarks.common import cached, fetch_diff, finish, linked_issues
from benchmarks.registry import Benchmark, get_benchmark
from benchmarks.schema import KnownIssue, PRExample, norm_severity, write_jsonl

FINDING_EXTRA = (
    "scope",
    "difficulty",
    "context_required",
    "producer",
    "tp_fp_justification",
    "severity_justification",
    "category_justification",
)


def fetch_sources(bench: Benchmark, refresh: bool = False) -> dict:
    """Download the pinned manifest, test subset and golden findings into raw/ (one tarball)."""
    cfg = bench.config
    raw = bench.raw_dir
    manifest_path = raw / "manifest.json"
    if refresh or not manifest_path.exists():
        url = f"https://codeload.github.com/{cfg['source_repo']}/tar.gz/{cfg['source_commit']}"
        data = github._request(url, accept="*/*")
        golden: dict[str, dict] = {}
        wanted = {cfg["files"]["manifest"]: "manifest.json", cfg["files"]["test_subset"]: "test_subset.json"}
        raw.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                rel = member.name.split("/", 1)[-1]  # strip "<repo>-<sha>/"
                if rel in wanted or (rel.startswith(cfg["files"]["golden_dir"] + "/") and rel.endswith(".json")):
                    content = json.loads(tar.extractfile(member).read())
                    if rel in wanted:
                        (raw / wanted[rel]).write_text(json.dumps(content))
                    else:
                        golden[content["pr_key"]] = content
        (raw / "golden.json").write_text(json.dumps(golden))
    return {
        "manifest": json.loads(manifest_path.read_text()),
        "test_subset": json.loads((raw / "test_subset.json").read_text()),
        "golden": json.loads((raw / "golden.json").read_text()),
    }


def pr_key(entry: dict) -> str:
    """ReviewBench's own key: <owner>_<repo>_<number>-<head[:8]>."""
    nwo = entry.get("nwo") or entry["repo"].removeprefix("https://github.com/").strip("/")
    return f"{nwo.replace('/', '_')}_{entry['pr_number']}-{entry['head'][:8]}"


def known_issues(ex_id: str, findings: list[dict]) -> list[KnownIssue]:
    out = []
    for i, f in enumerate(findings):
        out.append(
            KnownIssue(
                id=f"{ex_id}#{i}",
                body=f["message"].strip(),
                severity=norm_severity(f.get("severity")),
                category=f.get("category"),
                file=f.get("file"),
                start_line=f.get("start_line"),
                end_line=f.get("end_line", f.get("start_line")),
                label=f.get("tp_fp", "tp"),
                source=(f.get("source") or {}).get("type"),
                extra={k: f[k] for k in FINDING_EXTRA if k in f},
            )
        )
    return out


def convert_one(entry: dict, golden: dict, in_test_subset: bool, bench: Benchmark, refresh: bool = False) -> PRExample:
    key = pr_key(entry)
    if key not in golden:
        raise KeyError(f"no golden findings for {key}")
    nwo = entry["nwo"]
    mirror = f"{bench.config['mirror_org']}/{nwo.replace('/', '_')}"
    ex_id = f"reviewbench/{nwo.replace('/', '_')}_{entry['pr_number']}"
    pr = cached(
        bench.raw_dir / "prs" / f"{key}.json",
        lambda: github.pull_request(nwo, entry["pr_number"]) or {},
        refresh,
    )
    ex = PRExample(
        id=ex_id,
        benchmark="reviewbench",
        split=bench.split,
        repo=nwo,
        pr_number=entry["pr_number"],
        pr_url=entry["pr_url"],
        clone_url=f"https://github.com/{nwo}.git",
        base_sha=entry["base"],
        head_sha=entry["head"],
        title=entry.get("title") or "",
        description=entry.get("body") or "",
        diff="",
        known_issues=known_issues(ex_id, golden[key]["findings"]),
        metadata={
            "pr_key": key,
            "language": entry.get("language"),
            "repo_size_kb": entry.get("repo_size_kb"),
            "manifest_lines_added": entry.get("lines_added"),
            "manifest_lines_removed": entry.get("lines_removed"),
            "manifest_files_changed": entry.get("files_changed"),
            "mirror_url": f"https://github.com/{mirror}.git",
            "in_test_subset": in_test_subset,
            "upstream_pr_available": bool(pr),
        },
    )
    ex.linked_issues = linked_issues(nwo, pr or None)
    diff_path = bench.raw_dir / "diffs" / f"{key}.diff"
    if diff_path.exists() and not refresh:
        ex.diff = diff_path.read_text()
    else:
        ex.diff = fetch_diff(ex, [mirror, nwo])
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(ex.diff)
    return finish(ex, pr or None)


def download(refresh: bool = False, limit: int | None = None) -> int:
    bench = get_benchmark("reviewbench")
    src = fetch_sources(bench, refresh)
    test_keys = {pr_key(e) for e in src["test_subset"]}
    entries = sorted(src["manifest"], key=pr_key)
    examples, failed = [], []
    for i, entry in enumerate(entries[:limit]):
        try:
            examples.append(convert_one(entry, src["golden"], pr_key(entry) in test_keys, bench, refresh))
            print(f"[reviewbench] {i + 1}/{len(entries)} {examples[-1].id}", flush=True)
        except Exception as e:
            failed.append((entry["pr_url"], str(e)))
            print(f"[reviewbench] FAILED {entry['pr_url']}: {e}", flush=True)
    n = write_jsonl(examples, bench.prs_path)
    print(f"[reviewbench] wrote {n} PRs -> {bench.prs_path}")
    for url, err in failed:
        print(f"  failed: {url}: {err}")
    return n
