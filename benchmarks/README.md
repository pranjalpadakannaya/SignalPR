# Benchmarks in one PR format (issues #4, #5)

Every benchmark is converted into the same `PRExample` format, so the agent, the baselines and the
evaluation harness read every PR the same way.

| Benchmark | Split | PRs | Ground truth | Use it for |
|---|---|---|---|---|
| `martian`: [Martian Code Review Bench](https://github.com/withmartian/code-review-benchmark) (offline set) | **dev** | 50 PRs from 5 repos (Sentry, Grafana, Cal.com, Discourse, Keycloak) | 173 human-curated golden comments: severity + category, **no file/line** | Prompt and threshold tuning, ablations, debugging |
| `reviewbench`: [GitHub ReviewBench](https://github.com/review-bench/ReviewBench) (full set) | **test** | 219 PRs from 187 repos | 4,632 findings (2,623 TP, 2,009 judged FP) with file + line range, severity, category | Final evaluation of the frozen config **only** |

Both sources are pinned to a commit in [`benchmarks.yaml`](benchmarks.yaml).

## Quick start

```bash
pip install -r requirements.txt          # only pyyaml is needed for this package
gh auth login                            # or export GITHUB_TOKEN=... (GitHub API, 5000 req/h)

python -m benchmarks download martian reviewbench   # ~5 min, writes data/benchmarks/<name>/prs.jsonl
python -m benchmarks stats martian reviewbench      # counts + check against the published numbers
python -m benchmarks checkout martian               # repos/martian/<pr>/ at each PR's head commit
```

Downloads are cached per PR under `data/benchmarks/<name>/raw/`, so an interrupted run resumes where it
stopped (`--refresh` refetches; `--limit N` for a smoke test). `data/` and `repos/` are git-ignored.

## Load it

```python
from benchmarks import load_benchmark

for ex in load_benchmark("martian", purpose="tune"):   # DEV set
    ex.diff, ex.files, ex.description, ex.linked_issue_text
    ex.real_issues        # ground truth (label == "tp")

test = load_benchmark("reviewbench", purpose="eval")  # TEST set: final, frozen config only
load_benchmark("reviewbench", purpose="tune")         # raises TestSetAccessError
load_benchmark("reviewbench", purpose="eval", subset="test_subset")  # ReviewBench's 25-PR smoke-test slice
```

### The test-set guard

ReviewBench is marked `split: test` in `benchmarks.yaml`. `load_benchmark()` and `check_access()` refuse
any purpose other than `"eval"` for a test benchmark, so a tuning script that points at it fails right away.
Rules for the team:
- Tune prompts, thresholds and ablation choices on **Martian only**.
- Run ReviewBench only on a tagged frozen config (#11), and only through the eval harness (#7).
- Do not read ReviewBench findings to write prompts or rules.

## The `PRExample` format ([`schema.py`](schema.py))

| Field | Meaning |
|---|---|
| `id` | `<benchmark>/<owner>_<repo>_<pr>`, stable across re-downloads |
| `benchmark`, `split` | `martian`/`reviewbench`, `dev`/`test` |
| `repo`, `pr_number`, `pr_url` | the PR the benchmark points at |
| `clone_url`, `base_sha`, `head_sha` | a remote that holds both commits (ReviewBench: `metadata.mirror_url` is tried first) |
| `title`, `description` | PR title and body |
| `diff` | unified diff `base...head` |
| `files` | `ChangedFile(path, status, old_path, additions, deletions, binary)` |
| `linked_issues` | `LinkedIssue(number, title, body, url)`: issues the PR closes, or `fixes #N` references in its body |
| `known_issues` | `KnownIssue(id, body, severity, category, file, start_line, end_line, label, source, extra)` |
| `metadata` | `language`, `lines_added`, `lines_removed`, `files_changed`, `size_bucket`, `author`, `author_kind`, `agent`, plus benchmark-specific fields |

Notes:
- **Severity** is lower-cased: `critical | high | medium | low`. Martian uses all four. ReviewBench has no
  "critical" (its scale is high/medium/low). `schema.SEVERITY_WEIGHT` gives the weights for severity-weighted recall.
- **`label`**: Martian has only real issues (`tp`). ReviewBench also keeps judged false positives (`fp`):
  in its grounded precision, a comment that matches one of these counts as a false positive, so the harness needs them.
  Use `ex.real_issues` for recall.
- **Locations**: ReviewBench lines are on the head commit. Martian golden comments have no location, so
  Martian is matched on meaning only (the LLM judge in #7).
- **Martian categories** keep Martian's own scoring profiles in `extra["profiles"]`
  (`strict` 139 / `core` 158 / `all` 173 comments; `core` is Martian's default).
- **ReviewBench extras** per finding: `scope` (introduced/pre-existing), `difficulty`, `context_required`,
  `producer` and the labelers' justifications.
- **`size_bucket`** (added + removed lines): `xs` ≤50, `s` ≤200, `m` ≤500, `l` ≤1000, `xl` >1000, the same cut points ReviewBench reports.
- **`author_kind`** (`agent | human | bot | unknown`) and `agent` are inferred, because neither benchmark labels them
  ([`authorship.py`](authorship.py)). The signals are the PR author account (`Copilot`, `devin-ai-integration[bot]`, ...),
  the head branch (`codex/`, `cursor/`, `copilot/`, ...), agent markers in the body, and co-author trailers in commits.
  `unknown` means the upstream PR is gone, or the PR was re-created from a commit (Martian's Discourse PRs).

### Martian specifics
35 PRs live in the upstream repos. The other 15 were re-created in the `ai-code-review-evaluation` org,
and their `metadata.original_url` points at the upstream PR or commit. For those, the description comes
from the upstream PR body or commit message (the re-created PRs only say "Test N"), and authorship and
linked issues come from the upstream PR when there is one.

### ReviewBench specifics
- `diff` is the three-dot `base...head` diff, which is what ReviewBench's agent contract hands reviewers
  (`/work/pr/diff.patch`). For 57 PRs, the manifest's `lines_added`/`files_changed` disagree with that diff
  (the manifest seems to reflect the PR's state at another time). We keep them as `metadata.manifest_*`
  and use them only to reproduce ReviewBench's published size table. `metadata.size_bucket` comes from the real diff.
- 27 of the 2,623 TP findings point at files outside the diff (mostly `scope: pre-existing`).
- 11 upstream PRs no longer exist. Their commits still come from the mirror, but `author_kind` is `unknown`.
- `metadata.in_test_subset` marks ReviewBench's 25-PR smoke-test slice.

## Checking out repos

`python -m benchmarks checkout <name> [--id <pr id>] [--ref base]` fetches each repo once into a shared
blobless cache (`repos/.cache/<owner>_<repo>.git`), fetching only the needed commits, then creates a
`git worktree` per PR at `repos/<benchmark>/<owner>_<repo>_<pr>/` (`...@base` for the base commit).
From code: `benchmarks.checkout.checkout(ex, ref="head")` returns the path.

Each worktree is a full source tree (Sentry about 200 MB, Grafana and Keycloak larger), so all 50 Martian
PRs take on the order of 10 GB and 30+ minutes the first time. Prefer checking out on demand (`--id`, or
`checkout(ex)` inside the pipeline). Delete `repos/<benchmark>/` freely; the cache makes re-checkout fast.

## Tests

```bash
pytest tests/benchmarks      # offline: schema, guard, converters, authorship, stats, git checkout
```
