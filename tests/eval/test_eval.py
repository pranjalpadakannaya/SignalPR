import json

from agent.samples import SAMPLES_DIR
from benchmarks.schema import KnownIssue, LinkedIssue, PRExample, write_jsonl
from eval.__main__ import main as cli
from eval.judge import llm_match
from eval.metrics import aggregate, match, pr_metrics
from eval.run import load_items, run_benchmark, to_markdown

ISSUES = [
    {
        "file": "app/db.py",
        "start_line": 10,
        "end_line": 12,
        "severity": "high",
        "category": "security",
        "body": "SQL injection in query",
        "label": "tp",
    },
    {
        "file": "app/db.py",
        "start_line": 30,
        "severity": "low",
        "category": "maintainability",
        "body": "duplicated helper",
        "label": "tp",
    },
    {"file": "app/db.py", "start_line": 50, "severity": "low", "category": "style", "body": "rename x", "label": "fp"},
]


def c(file, line, title, category="correctness"):
    return {"file": file, "line": line, "title": title, "body": "", "category": category}


def test_match_ranges_and_order():
    comments = [
        c("app/db.py", 16, "SQL injection via f-string", "security"),  # 10-12 range + 5 tolerance
        c("app/db.py", 11, "another SQL injection note", "security"),  # issue already used
        c("app/other.py", 10, "SQL injection", "security"),  # wrong file
        c("db.py", 31, "duplicated helper code", "design"),  # partial path is fine
        c("app/db.py", 50, "rename x", "design"),  # matches a judged false positive
    ]
    assert match(comments, ISSUES) == [0, None, None, 1, 2]
    m = pr_metrics(comments, ISSUES)
    assert (m["tp"], m["n_gold"], m["matched_fp"], m["n_comments"], m["found@1"]) == (2, 2, 1, 5, 1)
    assert m["sev_found"] == m["sev_total"] == 4.0  # high 3 + low 1


def test_unlocated_issues_match_on_wording():
    issues = [{"file": None, "body": "Django querysets do not support negative slicing", "category": "bug"}]
    hit = {"file": "a.py", "line": 3, "title": "negative slicing", "body": "querysets reject negative slicing"}
    miss = {"file": "a.py", "line": 3, "title": "rename variable", "body": "unclear name", "category": "correctness"}
    assert match([miss, hit], issues) == [None, 0]


def test_aggregate_values_and_ci():
    rows = [pr_metrics([c("app/db.py", 10, "SQL injection", "security")], ISSUES), pr_metrics([], ISSUES)]
    a = aggregate(rows, n_boot=200)
    assert a["precision"] == 1.0 and a["recall"] == 0.25 and abs(a["f1"] - 0.4) < 1e-9
    assert a["comments_per_real_issue"] == 1.0 and a["recall@1"] == 0.25
    lo, hi = a["ci95"]["recall"]
    assert 0 <= lo <= a["recall"] <= hi <= 1


def test_llm_judge(monkeypatch):
    monkeypatch.setattr(
        "agent.llm.chat_json",
        lambda *a, **k: (
            {"matches": [{"comment": 1, "issue": 2}, {"comment": 2, "issue": 2}, {"comment": 3, "issue": None}]},
            "",
        ),
    )
    assert llm_match([{"title": "x"}, {"title": "y"}, {"title": "z"}], [{"body": "a"}, {"body": "b"}]) == [
        1,
        None,
        None,
    ]


def test_prexample_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNALPR_LLM", "off")
    diff = json.loads((SAMPLES_DIR / "flask-user-search.json").read_text())["diff"]
    ex = PRExample(
        id="martian/x_1",
        benchmark="martian",
        split="dev",
        repo="o/r",
        pr_number=1,
        pr_url="u",
        clone_url="c",
        base_sha="b",
        head_sha="h",
        title="Add user search",
        description="",
        diff=diff,
        linked_issues=[LinkedIssue(3, "Admin user search", "limit max 100")],
        known_issues=[
            KnownIssue(
                "k0",
                "SQL injection with f-string query",
                "critical",
                "security",
                extra={"profiles": ["strict", "core", "all"]},
            ),
            KnownIssue("k1", "style nit", "low", "style", extra={"profiles": ["all"]}),
        ],
        metadata={"language": "Python", "agent": "OpenAI_Codex", "size_bucket": "xs"},
    )
    path = tmp_path / "prs.jsonl"
    write_jsonl([ex], path)
    items = load_items(str(path), profile="core")
    assert len(items[0]["issues"]) == 1 and items[0]["pr"].issue_title == "Admin user search"
    res = run_benchmark(str(path), ["SignalPR (full)"], profile="core")
    assert res["judge"] == "lexical" and "warning" in res
    assert res["systems"]["SignalPR (full)"]["true_positives"] == 1


def test_demo_benchmark_runs_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNALPR_LLM", "off")
    out = tmp_path / "demo.json"
    res = run_benchmark("demo", ["SignalPR (full)", "Single-prompt LLM (same model)"], out=out)
    assert "Single-prompt LLM (same model)" in res["skipped"]
    full = res["systems"]["SignalPR (full)"]
    assert full["recall"] > 0.5 and full["precision"] > 0.5 and json.loads(out.read_text())["n_prs"] == 4
    assert "| SignalPR (full) |" in to_markdown(res)
    assert cli(["--bench", "demo", "--systems", "Code-analysis tools only", "--out", str(tmp_path / "x.json")]) == 0
    assert (tmp_path / "x.md").exists()
