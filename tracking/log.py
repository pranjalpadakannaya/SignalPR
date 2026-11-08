"""Experiment tracking (issue #3).

Every run gets its own folder, results/runs/<run_id>/, holding:
  config.yaml   what was run: benchmark, systems, model, prompt version, seed, commit, owner
  result.json   the raw output of the run (for example results/eval/<name>.json)
  result.md     the readable table, when there is one

and one row per system is added to the shared sheet results/runs/runs.tsv.
The .tsv file opens directly in Google Sheets or Excel (File -> Import).

The goal: any number in the final report can be traced to a run folder and a commit.
"""

from __future__ import annotations

import getpass
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "runs"
SHEET = RUNS_DIR / "runs.tsv"

# Columns of the shared results sheet, in order.
COLUMNS = [
    "run_id",
    "date",
    "system",
    "model",
    "prompt_version",
    "benchmark",
    "subset",
    "repeats",
    "seed",
    "judge",
    "n_prs",
    "precision",
    "recall",
    "f1",
    "sev_weighted_recall",
    "recall@1",
    "recall@3",
    "recall@5",
    "comments_per_real_issue",
    "commit",
    "owner",
    "notes",
]


def _git(*args: str) -> str:
    """Run a git command in the repo and return its output ("" if git is not available)."""
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_commit() -> str:
    """Short hash of the current commit, with '+dirty' if there are uncommitted changes."""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    if _git("status", "--porcelain", "--untracked-files=no"):
        commit += "+dirty"
    return commit


def default_owner() -> str:
    return _git("config", "user.name") or getpass.getuser()


def new_run_id(benchmark: str) -> str:
    """Readable, unique id such as 20261108-1903_martian_a1b2c3d."""
    stamp = time.strftime("%Y%m%d-%H%M")
    commit = git_commit().replace("+dirty", "d")
    run_id = f"{stamp}_{benchmark}_{commit}"
    n = 2
    while (RUNS_DIR / run_id).exists():  # two runs in the same minute
        run_id = f"{stamp}_{benchmark}_{commit}_{n}"
        n += 1
    return run_id


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\t", " ").replace("\n", " ")


def log_run(
    result: dict | str | Path,
    *,
    prompt_version: str = "inline",
    seed: int | None = 0,
    owner: str | None = None,
    notes: str = "",
    config: dict | None = None,
) -> Path:
    """Save one evaluation run and add it to the results sheet.

    result: the dict returned by eval.run.run_benchmark(), or the path to its JSON file
            (results/eval/<name>.json). Its .md table is copied too when it exists.
    prompt_version: which prompt files were used, e.g. "v1" (see prompts/README.md).
    config: any extra settings to record (for example the command-line options).

    Returns the run folder, results/runs/<run_id>/.
    """
    src = None
    if not isinstance(result, dict):
        src = Path(result)
        result = json.loads(src.read_text())

    benchmark = str(result.get("benchmark", "unknown"))
    model = (result.get("llm") or {}).get("model") or "none"
    owner = owner or default_owner()
    commit = git_commit()
    run_id = new_run_id(Path(benchmark).stem)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)

    # 1) what was run
    run_config = {
        "run_id": run_id,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "owner": owner,
        "commit": commit,
        "benchmark": benchmark,
        "subset": result.get("subset"),
        "profile": result.get("profile"),
        "model": model,
        "prompt_version": prompt_version,
        "seed": seed,
        "judge": result.get("judge"),
        "repeats": result.get("repeats"),
        "systems": list(result.get("systems", {})),
        "skipped": result.get("skipped", {}),
        "notes": notes,
        "extra": config or {},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True))

    # 2) the raw outputs
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    if src is not None and src.with_suffix(".md").exists():
        shutil.copy(src.with_suffix(".md"), run_dir / "result.md")

    # 3) one row per system in the shared sheet
    new_sheet = not SHEET.exists()
    with SHEET.open("a", encoding="utf-8") as f:
        if new_sheet:
            f.write("\t".join(COLUMNS) + "\n")
        for system, m in result.get("systems", {}).items():
            row = {
                **run_config,
                "system": system,
                "n_prs": result.get("n_prs"),
                **{k: m.get(k) for k in COLUMNS if k in m},
            }
            f.write("\t".join(_fmt(row.get(c)) for c in COLUMNS) + "\n")
    return run_dir


def list_runs() -> list[dict]:
    """Rows of the results sheet as dicts (empty list if nothing is logged yet)."""
    if not SHEET.exists():
        return []
    lines = SHEET.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]
