# eval/ — evaluation harness (#7)

Runs SignalPR, the baselines and every ablation on the same PRs with the same model, and scores the
comments against each benchmark's known issues. PRs come from `benchmarks.load_benchmark()` (#4, #5).

```bash
python -m benchmarks download martian reviewbench       # once (see benchmarks/README.md)

python -m eval --bench demo                              # 4 demo PRs, no model needed: harness check only
python -m eval --bench martian --repeats 3               # DEV set; tune prompts and settings here
python -m eval --bench martian --profile core            # Martian's default scoring profile
python -m eval --bench reviewbench --subset test_subset  # TEST set (25-PR slice); frozen config only
python -m eval --bench reviewbench --repeats 3           # TEST set, all 219 PRs
```

Results go to `results/eval/<name>.json` and `.md`. Martian is loaded with `purpose="tune"`,
ReviewBench with `purpose="eval"`, so the test-set guard in `benchmarks/registry.py` applies.

| System | What changes |
|---|---|
| SignalPR (full) | all five steps |
| Single-prompt LLM (same model) | one prompt, no tools, verifier or ranker |
| Code-analysis tools only | analyzer findings as comments, ranked by severity |
| SignalPR, ranked by change size | ranking by file churn instead of the risk model |
| – intent checker / – evidence collector / – verifier / – self-consistency / – risk ranker | one step removed |

## Metrics (`metrics.py`)

- **Precision (grounded):** comments that match a real known issue / all comments. Matching one of
  ReviewBench's judged false positives, or nothing, counts against it.
- **Recall, F1** over real issues (`label == "tp"`), with 95% bootstrap CIs resampling PRs.
- **Severity-weighted recall** with `benchmarks.schema.SEVERITY_WEIGHT` (critical 4, high 3, medium 2, low 1).
- **Recall@1/3/5:** real issues found in the first k comments (isolates the ranker).
- **Comments per real issue:** how many comments a person reads to find one real problem.

Every per-PR row also keeps `language`, `size_bucket` and `author_kind`, for breakdowns by PR size,
language and agent- vs human-written PRs.

## Matching

A comment matches a known issue in the same file whose line range (±5 lines) contains the comment's
line, when its wording or category fits; each issue is matched at most once, in the order comments are
shown. Martian's golden comments have no location, so `--judge auto` (default) uses the LLM judge
(`judge.py`, one cached call per PR) whenever issues lack a file and a model is reachable; keep the
judge model fixed across systems.
