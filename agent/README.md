# agent/ — the SignalPR reviewer

Five steps, run as a LangGraph graph when `langgraph` is installed and as the same functions in
order otherwise:

| Step | File | What it does |
|---|---|---|
| 1. Intent checker | `intent.py` | Turns the issue + PR description into a checklist; marks each item covered / partial / missing and lists changed files nobody asked for |
| 2. Evidence collector | `evidence.py` | Ruff, ESLint, Semgrep, Lizard and built-in rules on changed lines (`tools/`), duplicated blocks, new definitions and where they are used, the project's tests |
| 3. Specialist reviewers | `specialists.py` | Correctness, security and design reviewers write candidate comments (LLM, several samples each); tool findings also become candidates |
| 4. Verifier | `verifier.py` | Keeps a comment only if it is on a changed line, is backed by a tool finding or a quote that is really on that line, and (LLM comments) appears in most samples; merges duplicates; records why each dropped comment was dropped |
| 5. Risk ranker | `ranker.py` | Scores comments by file risk from the risk model (`risk/`, issue #10), severity and evidence; riskiest first |

`pipeline.py` wires the steps and the baselines; `state.py` holds the data types; `llm.py` shares one
cached `LLMClient` (issue #2); `github.py` fetches PRs and posts reviews.

## Run

```bash
python -m agent samples
python -m agent review --sample flask-user-search            # demo PR with 8 known issues
python -m agent review --diff change.diff --title "Add search" --issue issue.md
python -m agent review --url https://github.com/owner/repo/pull/123   # GITHUB_TOKEN for private repos
python -m agent review --sample cart-rounding --mode single_prompt    # baselines: single_prompt | tools_only | size_rank
python -m agent review --sample cart-rounding --no-verifier           # ablations: --no-intent --no-evidence
                                                                     #   --no-verifier --no-consistency --no-ranker
```

Point it at a model server first (see `scripts/serve/README.md`), e.g.

```bash
export SIGNALPR_LLM_PRESET=qwen-7b SIGNALPR_LLM_BACKEND=ollama
python -m agent health
```

With no server reachable (or `SIGNALPR_LLM=off`) the review runs on tool evidence only. Every LLM
reply is cached, so re-running a review or an experiment gives the same output without new calls.

```python
from agent.pipeline import review
from agent.state import PRInput

result = review(PRInput.from_diff(diff_text, title="...", issue_title="...", issue_body="..."),
                {"mode": "signalpr", "runs": 3})
result["comments"]   # ranked, verified comments (file, line, severity, body, suggestion, score, ...)
result["dropped"]    # filtered candidates with drop_reason
result["intent"], result["evidence"], result["file_risk"], result["trace"]
```
