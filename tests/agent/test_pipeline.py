import json

import pytest

from agent import llm, pipeline, verifier
from agent.__main__ import main as cli
from agent.samples import list_samples, load_sample, pr_from_dict
from agent.state import Comment, Evidence, PRInput

DIFF = """diff --git a/app/search.py b/app/search.py
new file mode 100644
--- /dev/null
+++ b/app/search.py
@@ -0,0 +1,9 @@
+import requests
+
+
+def search(db, q, limit=20):
+    rows = db.execute(f"SELECT * FROM users WHERE name LIKE '%{q}%'")
+    try:
+        return rows.fetchall()[:limit]
+    except:
+        return []
"""


def offline(monkeypatch):
    monkeypatch.setattr(llm, "status", lambda force=False: {"available": False, "reason": "test", "model": None})


def fake_llm(monkeypatch, reply):
    """Route every LLM call to `reply(system, user, sample)`."""
    calls = []
    monkeypatch.setattr(llm, "status", lambda force=False: {"available": True, "reason": "", "model": "fake"})

    def chat_json(system, user, *, temperature=0.2, sample=0, tag=None, max_tokens=None):
        calls.append({"tag": tag, "sample": sample})
        data = reply(system, user, sample)
        return data, json.dumps(data)

    monkeypatch.setattr(llm, "chat_json", chat_json)
    return calls


def test_samples_load():
    ids = [s["id"] for s in list_samples()]
    assert "flask-user-search" in ids
    pr = pr_from_dict(load_sample("flask-user-search"))
    assert pr.files and pr.has_issue


def test_offline_review_finds_planted_issues(monkeypatch):
    offline(monkeypatch)
    pr = pr_from_dict(load_sample("flask-user-search"))
    res = pipeline.review(pr)
    found = {(c["file"], c["line"]) for c in res["comments"] + res["more_comments"]}
    gold = load_sample("flask-user-search")["gold"]
    hits = sum(any(f == g["file"] and abs((ln or 0) - g["line"]) <= 1 for f, ln in found) for g in gold)
    assert hits >= 5
    assert [t["step"] for t in res["trace"]] == ["intent", "evidence", "specialists", "verifier", "ranker"]
    assert res["comments"][0]["severity"] == "critical"
    scores = [c["score"] for c in res["comments"]]
    assert scores == sorted(scores, reverse=True)
    assert res["intent"]["requirements"] and any(r["status"] == "missing" for r in res["intent"]["requirements"])


def test_llm_comments_are_verified(monkeypatch):
    def reply(system, user, sample):
        if "intent checker" in system:
            return {
                "requirements": [{"text": "search users by name", "status": "covered", "files": ["app/search.py"]}],
                "unrequested": [],
                "summary": "does what was asked",
            }
        if "correctness specialist" not in system:
            return {"comments": []}
        good = {
            "file": "app/search.py",
            "line": 7,
            "severity": "medium",
            "title": "limit is applied after fetchall",
            "body": "All rows are loaded before slicing; pass LIMIT to the query.",
            "quote": "return rows.fetchall()[:limit]",
            "suggestion": "Add LIMIT to the SQL.",
        }
        hallucinated = {
            "file": "app/search.py",
            "line": 4,
            "severity": "critical",
            "title": "missing auth",
            "body": "`check_admin` is never called",
            "quote": "if user.is_admin:",
        }
        once = {
            "file": "app/search.py",
            "line": 5,
            "severity": "low",
            "title": "rename q",
            "body": "use a clearer name than q",
            "quote": "rows = db.execute",
        }
        return {"comments": [good, hallucinated] + ([once] if sample == 0 else [])}

    calls = fake_llm(monkeypatch, reply)
    res = pipeline.review(PRInput.from_diff(DIFF, title="Add user search"), {"runs": 3})
    assert res["llm"]["used"]
    titles = [c["title"] for c in res["comments"]]
    assert "limit is applied after fetchall" in titles
    reasons = {c["title"]: c["drop_reason"] for c in res["dropped"]}
    assert "hallucinated" in reasons["missing auth"]
    assert "1/3 runs" in reasons["rename q"]
    kept = next(c for c in res["comments"] if c["title"] == "limit is applied after fetchall")
    assert "3/3 runs agree" in kept["verified_by"]
    # three samples per specialist, each with its own sample index (separate cache entries)
    spec = [c for c in calls if (c["tag"] or "").startswith("specialist/correctness")]
    assert sorted(c["sample"] for c in spec) == [0, 1, 2]


def test_consistency_ablation_keeps_single_run_comments(monkeypatch):
    def reply(system, user, sample):
        if "correctness specialist" in system and sample == 0:
            return {
                "comments": [
                    {
                        "file": "app/search.py",
                        "line": 5,
                        "severity": "low",
                        "title": "rename q",
                        "body": "use a clearer name",
                        "quote": "rows = db.execute",
                    }
                ]
            }
        return {"comments": []} if "specialist" in system else {"requirements": []}

    fake_llm(monkeypatch, reply)
    pr = PRInput.from_diff(DIFF)
    on = pipeline.review(pr, {"runs": 3})
    off = pipeline.review(PRInput.from_diff(DIFF), {"runs": 3, "consistency": False})
    assert "rename q" not in [c["title"] for c in on["comments"] + on["more_comments"]]
    assert "rename q" in [c["title"] for c in off["comments"] + off["more_comments"]]


def test_baseline_modes(monkeypatch):
    def reply(system, user, sample):
        return {
            "comments": [
                {
                    "file": "app/search.py",
                    "line": 5,
                    "severity": "critical",
                    "title": "SQL injection",
                    "body": "f-string SQL",
                    "quote": "db.execute",
                }
            ]
        }

    fake_llm(monkeypatch, reply)
    sp = pipeline.review(PRInput.from_diff(DIFF), {"mode": "single_prompt"})
    assert [c["title"] for c in sp["comments"]] == ["SQL injection"] and sp["trace"][0]["step"] == "single prompt"
    offline(monkeypatch)
    tools = pipeline.review(PRInput.from_diff(DIFF), {"mode": "tools_only"})
    assert tools["comments"] and all(c["source"].endswith("/tools") for c in tools["comments"])
    size = pipeline.review(PRInput.from_diff(DIFF), {"mode": "size_rank"})
    assert all(c["score"] == 9.0 for c in size["comments"])  # 9 added lines in the only file
    with pytest.raises(RuntimeError):
        pipeline.review(PRInput.from_diff(DIFF), {"mode": "single_prompt"})
    with pytest.raises(ValueError):
        pipeline.Options.from_dict({"mode": "nope"})


def test_verifier_rules():
    pr = PRInput.from_diff(DIFF)
    ev = [Evidence("E1", "app/search.py", 8, "rules", "cor.bare_except", "correctness", "medium", "bare except")]
    cands = [
        Comment(
            "app/search.py",
            8,
            "correctness",
            "medium",
            "bare except",
            "catch Exception",
            "correctness/tools",
            evidence=["E1"],
        ),
        Comment("other.py", 1, "correctness", "low", "x", "y", "correctness/llm", quote="z"),
        Comment("app/search.py", 40, "correctness", "low", "x", "y", "correctness/llm"),
        Comment(
            "b/app/search.py",
            9,
            "correctness",
            "medium",
            "returns empty list",
            "hides errors",
            "correctness/llm",
            quote="return []",
        ),
    ]
    res = verifier.verify(pr, cands, ev, runs=1)
    kept = {(c.file, c.line) for c in res["kept"]}
    assert ("app/search.py", 8) in kept and ("app/search.py", 9) in kept
    reasons = sorted(c.drop_reason for c in res["dropped"])
    assert any("not part of this diff" in r for r in reasons) and any("not a changed line" in r for r in reasons)


def test_cli(monkeypatch, tmp_path, capsys):
    offline(monkeypatch)
    out = tmp_path / "r.json"
    d = tmp_path / "c.diff"
    d.write_text(DIFF)
    assert cli(["review", "--diff", str(d), "--title", "search", "--no-ranker", "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["options"]["ranker"] is False and res["comments"]
    assert "comment(s)" in capsys.readouterr().out
