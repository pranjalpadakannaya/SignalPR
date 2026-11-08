# tracking/ — experiment tracking (issue #3)

Goal: **any number in the final report can be traced to a run folder and a commit.**

## How to log a run

1. Commit your code first, so the run points to a real commit (otherwise the commit is marked `+dirty`).
2. Run the evaluation as usual:

   ```bash
   python -m eval --bench martian --repeats 3
   ```

3. Log it:

   ```bash
   python -m tracking log results/eval/martian.json --prompt-version v1 --notes "first Martian run, 7B"
   ```

   This creates `results/runs/<run_id>/` and adds one row per system to `results/runs/runs.tsv`.

4. Commit the run folder and the sheet:

   ```bash
   git add results/runs
   git commit -m "Log run <run_id>"
   ```

5. See everything logged so far:

   ```bash
   python -m tracking list
   ```

From Python, `from tracking import log_run; log_run(res, prompt_version="v1", notes="...")` takes the
dict returned by `eval.run.run_benchmark()` directly.

## What a run folder holds

| File | Contents |
|---|---|
| `config.yaml` | run id, date, owner, commit, benchmark, subset, model, prompt version, seed, judge, repeats, systems run and skipped, notes |
| `result.json` | the raw output of `python -m eval` (all metrics and per-PR rows) |
| `result.md` | the readable results table, when there is one |

## The shared results sheet

`results/runs/runs.tsv` has one row per system per run, with the columns: run id, date, system, model,
prompt version, benchmark, subset, repeats, seed, judge, number of PRs, precision, recall, F1,
severity-weighted recall, recall@1/3/5, comments per real issue, commit, owner, and notes.

To view it as a spreadsheet: Google Sheets → File → Import → Upload `runs.tsv` (separator: tab).
It is a `.tsv` file because the repo's `.gitignore` excludes `*.csv`.

## Rules

- Report a number in the paper only if its run is logged here.
- Use Martian (dev) for tuning; log ReviewBench (test) runs only with a frozen config.
- Never edit a logged run. If a run was wrong, log a new one and say why in `--notes`.

## Still to do (after #19 and #20 are merged)

- Move the agent's prompts into versioned files (`prompts/<step>/v1.txt`, see `prompts/README.md`).
- Call `log_run()` automatically at the end of `python -m eval`.
