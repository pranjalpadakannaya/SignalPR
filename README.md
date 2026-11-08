# SignalPR
A Code Review Agent That Learns from Real Reviewers Which Comments Matter

SignalPR reviews pull requests with an LLM pipeline: it checks the PR against its intent, gathers
evidence (tests, static tools, call sites), runs specialist reviewers, verifies each comment, and
ranks the result by file risk, learned from what real reviewers comment on. It is evaluated against
single-prompt and PR-Agent baselines on two code-review benchmarks.

## Repository layout

| Path | What it is | Issue |
|---|---|---|
| `benchmarks/` | Martian (dev) and ReviewBench (test) loaded into one `PRExample` format, repo checkout | #4, #5 |
| `risk/` | File-level risk model trained on AIDev: which files reviewers comment on | #10 |
| `tools/` | Shared static-analysis runners (Ruff, ESLint, Semgrep, Lizard, built-in rules) and a diff parser | #10 |
| `agent/` | Shared LLM client (`llm_client.py`) and model presets; the agent pipeline goes here | #2, #11 |
| `scripts/serve/` | Serving Qwen-Coder with vLLM (ASU Sol) or Ollama (laptop / Kaggle) | #2 |
| `eval/` | Evaluation harness (planned) | #7 |
| `app/` | GitHub App and dashboard (planned) | #13 |
| `notebooks/` | Kaggle notebook for serving the model | #2 |
| `results/` | Committed metrics and figures | |
| `tests/` | `pytest` suite, one folder per package | |

Not committed (git-ignored): `data/` (datasets), `repos/` (checked-out benchmark repos), `models/`
(trained models), `.cache/` (LLM response cache), `.venv/`.

## Setup

Requires **Python 3.10–3.12** (the pinned numpy 1.26 has no wheels for newer versions), git, and the
GitHub CLI.

```bash
git clone https://github.com/pranjalpadakannaya/SignalPR.git && cd SignalPR
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
gh auth login        # or: export GITHUB_TOKEN=...   (needed to download the benchmarks)
```

macOS only: LightGBM needs OpenMP (`brew install libomp`). Without it, the risk model falls back to
scikit-learn.

Run every command from the repository root.

## Quick start

**Benchmarks** ([benchmarks/README.md](benchmarks/README.md))
```bash
python -m benchmarks download martian reviewbench   # ~5 min → data/benchmarks/
python -m benchmarks stats martian reviewbench      # check counts against the published numbers
python -m benchmarks checkout martian --id <pr id>  # repo at the PR's head → repos/
```
```python
from benchmarks import load_benchmark
dev = load_benchmark("martian", purpose="tune")       # tune on this
test = load_benchmark("reviewbench", purpose="eval")  # final evaluation only
```

**Risk model and tool runners** ([risk/README.md](risk/README.md))
```bash
python -m risk download && python -m risk mine && python -m risk train && python -m risk report
```
Or copy `models/risk_model.joblib` from the team drive and call `risk.predict_risk(files, pr)`.

**LLM serving and client** ([scripts/serve/README.md](scripts/serve/README.md))
```bash
bash scripts/serve/ollama_local.sh          # laptop, Qwen2.5-Coder 7B
sbatch scripts/serve/sol_vllm.sbatch        # ASU Sol, vLLM
```
```python
from agent.llm_client import LLMClient, LLMConfig
llm = LLMClient(LLMConfig.from_preset("qwen-7b", backend="ollama"))
```

## Tests

```bash
python -m pytest -q tests                   # everything (offline; skips what needs a model)
python -m pytest -q tests/benchmarks        # one package
```

## Working on the project

- **Branches:** each person works on `dev/<name>`. Merge `origin/main` into it before starting, and
  open a PR into `main` when a piece of work is done.
- **Issues:** every task is a GitHub issue (#1–#13). Reference it in commit messages, e.g. `(#4)`.
- **Dev vs test data:** tune prompts, thresholds and ablations on **Martian only**. Run **ReviewBench**
  only on a frozen, tagged config through the eval harness. The loader refuses
  `load_benchmark("reviewbench", purpose="tune")`.
- **Large files:** keep datasets, models and repo checkouts out of git. Share them through the team drive.
- **Style:** `ruff check .` and `ruff format --line-length 120`.
