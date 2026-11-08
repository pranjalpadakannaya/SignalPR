import json

import yaml

import tracking.log as tl

FAKE_RESULT = {
    "benchmark": "demo",
    "subset": None,
    "profile": None,
    "n_prs": 4,
    "llm": {"model": "qwen2.5-coder:7b"},
    "judge": "lexical",
    "repeats": 1,
    "systems": {
        "SignalPR (full)": {"precision": 0.71, "recall": 0.79, "f1": 0.75, "recall@3": 0.42},
        "Code-analysis tools only": {"precision": 0.5, "recall": 0.3, "f1": 0.375},
    },
    "skipped": {},
}


def _use_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(tl, "SHEET", tmp_path / "runs" / "runs.tsv")


def test_log_run_writes_folder_and_sheet(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    run_dir = tl.log_run(FAKE_RESULT, prompt_version="v1", owner="tester", notes="unit test")

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["model"] == "qwen2.5-coder:7b"
    assert cfg["prompt_version"] == "v1"
    assert cfg["owner"] == "tester"
    assert cfg["commit"]
    assert json.loads((run_dir / "result.json").read_text())["n_prs"] == 4

    rows = tl.list_runs()
    assert [r["system"] for r in rows] == ["SignalPR (full)", "Code-analysis tools only"]
    assert rows[0]["precision"] == "0.7100"
    assert rows[0]["run_id"] == run_dir.name
    assert list(rows[0]) == tl.COLUMNS


def test_log_from_file_copies_markdown_and_ids_are_unique(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    src = tmp_path / "demo.json"
    src.write_text(json.dumps(FAKE_RESULT))
    src.with_suffix(".md").write_text("| table |")

    a = tl.log_run(src, owner="tester")
    b = tl.log_run(src, owner="tester")
    assert a != b
    assert (a / "result.md").read_text() == "| table |"
    assert len(tl.list_runs()) == 4  # two runs x two systems, one header line only


def test_no_model_is_recorded_as_none(tmp_path, monkeypatch):
    _use_tmp(tmp_path, monkeypatch)
    run_dir = tl.log_run({**FAKE_RESULT, "llm": None}, owner="tester")
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["model"] == "none"
