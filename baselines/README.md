# baselines/ — PR-Agent (issue #9)

The open-source [Qodo PR-Agent](https://github.com/qodo-ai/pr-agent) `review` command, run **offline** on a
PR's diff (its `plain-diff` mode, no GitHub access) against the same Qwen server as every other system.
Its `key_issues_to_review` are converted to our `Comment` format (`baselines/pr_agent.py`), so it is scored
by the same `eval/` harness and logged by `tracking/`.

## One-time setup

PR-Agent needs Python ≥ 3.12 and pins its own `litellm`, and the PyPI release (0.2.4) is too old for
plain-diff mode, so install it from GitHub into its **own** virtualenv, not the project one:

```bash
python3.13 -m venv ~/.venvs/pr-agent
~/.venvs/pr-agent/bin/pip install "git+https://github.com/qodo-ai/pr-agent.git"
export SIGNALPR_PR_AGENT_BIN=~/.venvs/pr-agent/bin/pr-agent
```

## Run (Martian, 7B, laptop)

```bash
bash scripts/serve/ollama_local.sh
export SIGNALPR_LLM_PRESET=qwen-7b SIGNALPR_LLM_BACKEND=ollama
python -m eval --bench martian --profile core --systems "PR-Agent (open source)" \
    --out results/eval/martian_pr_agent_7b.json
python -m tracking log results/eval/martian_pr_agent_7b.json --prompt-version pr-agent-0.49 --notes "#9, 7B"
```

For 32B on Kaggle or Sol, serve the model (`scripts/serve/README.md`) and set `SIGNALPR_LLM_PRESET=qwen-32b`
plus `SIGNALPR_LLM_BACKEND` / `SIGNALPR_LLM_BASE_URL` for that server; nothing else changes. Use the same
model for the judge and the reviewer, and compare systems only within one model.

## Notes

- PR-Agent keeps its own prompts, so there is no `prompts/` file; log its version as `--prompt-version pr-agent-<version>`.
- Replies are cached in `.cache/pr_agent/` (keyed by diff, model and settings); delete it or set
  `SIGNALPR_LLM_CACHE=refresh` to re-run.
- A PR where PR-Agent fails (bad YAML from a small model, timeout) scores as no comments and the error is
  kept in the run output, like the single-prompt baseline.
- PR-Agent sees only the patch (no repo checkout), the same view SignalPR's single-prompt baseline gets.
- ReviewBench is run at final-run time with the frozen config (`--bench reviewbench`).
