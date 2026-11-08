# Risk model + shared tool runners (issue #10)

Owner: Sree Sai Harshitha Nanjyala (data & analysis tools)

This package answers one question: **which changed files in an agent-written PR will a human
reviewer comment on?** The agent's ranker (#11) uses the answer to show the most important comments
first. It also contains the shared code-analysis tool runners (`tools/`), which the agent
and the baselines reuse.

## Use it from other modules

```python
# 1) shared tool runners: Ruff, ESLint, Semgrep, Lizard + built-in rules, on changed lines, cached
from tools import files_from_diff, run_tools

report = run_tools(files_from_diff(diff_text))  # ToolReport
report.findings  # [Finding(file, line, tool, rule, category, severity, message), ...]
report.status  # {"ruff": "ran on 2 file(s)", "semgrep": "not installed", ...}
report.counts(path)  # {"tool_warnings": 3, "tool_security": 1, ...} per file

# 2) risk of each changed file (calibrated probability that a human comments on it)
from risk import predict_risk

risks = predict_risk(
    files=[
        {
            "filename": "app/routes/search.py",
            "status": "added",
            "additions": 30,
            "deletions": 0,
            "patch": "@@ ... (optional)",
        }
    ],
    pr={
        "agent": "OpenAI_Codex",
        "title": "...",
        "body": "...",
        "issue_text": "...",
        "stars": 640,
        "forks": 88,
        "repo_language": "Python",
    },
)
# -> [{"file", "risk", "risk_rel" (0..1 within the PR), "rank", "additions", "deletions", "lang", "source"}]
```

`files` can also be `tools.diff.FileDiff` objects from `parse_unified_diff()`.
Run everything from the repo root (`python -m risk ...`, `from risk import predict_risk`).

### Getting the trained model

Model files are git-ignored (repo policy: `models/`, `*.joblib`). The model lives at
`models/risk_model.joblib`. Either:
- **Copy it from the team drive** into `models/` (no data needed), or
- **Train it yourself** (below). It takes about 5 minutes once the data is in `data/raw/`.

If the file is missing, `predict_risk()` still works but falls back to ranking by change size, and
the `source` field says so. A LightGBM model is saved in a version-independent format (model text and
calibration as plain lists), so it loads with any scikit-learn version.

## Reproduce (about 5 minutes after the download)

```bash
pip install -r requirements.txt
# get AIDev-pop (hao-li/AIDev v5) into data/raw/ — either:
python -m risk download                # parquet, ~2-3 GB of traffic (no patch text)
python -m risk download --with-patch   # + patch features; streams the 12 GB patch column once
# or copy the *.csv.gz produced by scripts/prepare_aidev_csv.py into data/raw/

python -m risk mine      # clean + label + split + features  → data/processed/files.pkl
python -m risk train     # model → models/, metrics → results/risk/
python -m risk report    # figures (PDF+PNG) + summary.md     → results/risk/figures/
python -m pytest -q tests         # 15 tests
```

LightGBM is used when it imports, which is what the proposal specifies. Otherwise the code falls
back to scikit-learn's `HistGradientBoostingClassifier`, the same family of histogram gradient-boosted
trees. The training log and `metrics.json → backend` say which one was used. The pinned requirements
(numpy 1.26) need Python 3.10–3.12; on macOS, LightGBM also needs `libomp` (conda:
`conda install -c conda-forge libomp`, or Homebrew).

## Issue #10 checklist → where it lives

| Task | Code |
|---|---|
| Shared tool runners on changed lines, cached | `tools/runners.py`: `run_tools()`. The cache key is tool version + file content + changed lines (`~/.cache/signalpr-tools`, `SIGNALPR_TOOL_CACHE`) |
| Clean AIDev-pop | `risk/mine.py`: joins 7 tables; drops bot reviewers/comments and the author's own comments, empty comments, PRs > 300 files or files > 20k lines, open/unmerged PRs, PRs with no human review |
| File-level label, 70/10/20 split by repo and time | `mine.py`: label = a non-empty human inline comment on the file. Repos are hashed 70/10/20; train/val use PRs before the 80th-percentile date, test uses unseen repos after it |
| Features | `risk/features.py`: `build_features()` (58 active + 8 patch features). See below |
| Train LightGBM, report PR-AUC, calibration, importance on unseen repos | `model.py` (isotonic calibration on val, cluster-bootstrap CIs over PRs, baselines, permutation importance), `report.py` |
| Save model + `predict_risk()` | `models/risk_model.joblib` (shared, not committed), `risk/predict.py` |

### Features (issue #10 list → columns)

| Group | Columns |
|---|---|
| lines +/- | `additions`, `deletions`, `changes`, `log_changes`, `add_ratio`, `churn_share`, `churn_rank_pct`, `is_largest_file` |
| file counts (PR context) | `pr_n_files`, `pr_n_code_files`, `pr_n_dirs`, `pr_n_langs`, `same_dir_files`, `pr_additions`, `pr_deletions`, `pr_test_share`, `pr_has_tests` |
| complexity change | `complexity_delta` (branch tokens added − removed), `branches_added`, `n_hunks` |
| repeated code | `repeated_added_lines`, `repeated_ratio` |
| tool warnings | `tool_warnings`, `tool_security`, `tool_critical` (the same rules the tool runners use) |
| issue alignment | `issue_path_overlap`, `issue_path_hits` (file-path tokens vs linked-issue text, else PR title+body), `pr_issue_overlap` |
| agent name | `agent` (categorical) |
| also | path semantics (`is_test`, `is_doc`, `is_ci`, `is_lock`, `is_generated`, `tok_api`, `tok_auth`, …), `lang`, `status`, `task_type`, repo stars/forks/language |

Complexity change, repeated code and tool warnings come from the diff text. The committed model was
trained on the light download (no `patch` column), so those 8 columns were empty in training and were
skipped (`metrics.json → skipped_features`). `predict_risk()` still computes them; a model trained on
`download --with-patch` data will use them.

Leakage guard: per-file commit counts are not used, because AIDev counts commits pushed after the
review (fixes made in response to comments). Caveat: additions and deletions also include
post-review commits, because these tables don't timestamp individual commits.

## Results (test = 675 unseen repos, PRs after 2026-03-07)

| Ranker | PR-AUC (95% CI) | ROC-AUC |
|---|---|---:|
| **Risk model** | **0.267** (0.225–0.310) | **0.956** |
| Logistic regression, same numeric features | 0.086 (0.075–0.104) | 0.881 |
| Change size (lines changed) | 0.015 (0.013–0.019) | 0.649 |
| Random (base rate) | 0.010 | 0.48 |

That is 25× lift over the base rate, with calibration error (ECE) 0.003. Within a PR (347 test PRs with ≥ 2 files and ≥ 1 comment), the top-ranked file
drew a human comment **62%** of the time, vs 47% for change size and 29% for random. Adding the
issue-alignment features raised PR-AUC from 0.255 to 0.267. Full numbers: `results/risk/summary.md`.
Figures: `results/risk/figures/` (`pr_curve`, `calibration`, `feature_importance`, `within_pr_ranking`,
`pr_auc_baselines`, `rate_by_change_size`, `rate_by_file_category`, `merge_rate_by_agent`).
