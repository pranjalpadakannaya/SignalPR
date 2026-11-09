"""Intent checker, specialists and verifier: requirement comments, anchoring and duplicate handling."""

from agent import intent, specialists, verifier
from agent.state import Comment, Evidence, PRInput
from tests.agent.test_pipeline import fake_llm

DIFF = """diff --git a/app/search.py b/app/search.py
new file mode 100644
--- /dev/null
+++ b/app/search.py
@@ -0,0 +1,16 @@
+import sqlite3
+
+
+def get_db():
+    return sqlite3.connect("app.db")
+
+
+@bp.route("/search")
+def search_users():
+    q = request.args.get("q", "")
+    limit = int(request.args.get("limit", 20))
+    price = float(row[1])
+    db = get_db()
+    total = float(row[2])
+    rows = db.execute("SELECT * FROM users LIMIT ?", (limit,)).fetchall()
+    return rows
"""


def pr(**meta):
    return PRInput.from_diff(DIFF, **meta)


def test_requirement_contained_in_another_is_dropped():
    p = pr(issue_title="Admin page needs user search", issue_body="The admin page should let admins search users.")
    assert intent.extract_requirements(p) == ["The admin page should let admins search users."]


def test_best_line_ignores_generic_code_words():
    p = pr()
    assert intent.best_line(p, "Return 400 when the query is empty") is None  # only `return` is shared
    assert intent.best_line(p, "Cap the limit parameter at 100") == ("app/search.py", 11)


def test_unmet_requirements_become_anchored_comments():
    reqs = [
        {"id": "R1", "text": "Cap the limit parameter at 100", "status": "partial", "missing_terms": ["cap"]},
        {"id": "R2", "text": "Limit must default to 20 for each search", "status": "missing", "missing_terms": []},
        {"id": "R3", "text": "Add tests for the endpoint", "status": "missing"},  # left to the no-tests finding
        {"id": "R4", "text": "Send a weekly digest email", "status": "missing"},  # nothing related to anchor on
        {"id": "R5", "text": "Search users by name", "status": "covered"},
    ]
    out = specialists.requirement_comments(pr(), {"requirements": reqs})
    assert len(out) == 1  # R1 and R2 land on the same line and are folded together
    c = out[0]
    assert (c.file, c.line, c.category, c.severity) == ("app/search.py", 11, "intent", "medium")
    assert c.evidence == ["intent:R1", "intent:R2"] and "[partial] Cap the limit" in c.body and "`cap`" in c.body
    kept = verifier.verify(pr(), out, [], runs=1)["kept"]
    assert kept == [c] and "tool evidence" in c.verified_by


def test_file_level_finding_is_anchored_on_the_main_function_and_survives_the_verifier():
    p = pr()
    assert specialists.default_line(p.files[0]) == 9  # search_users (lines 9-16), not get_db or the import
    ev = [Evidence("E1", "app/search.py", 1, "tests", "no-tests", "tests", "medium", "16 lines, no tests.", "import")]
    cands = specialists.tool_comments("design", p, ev, {"requirements": [], "unrequested": []})
    assert [(c.line, c.evidence) for c in cands] == [(9, ["E1"])]
    assert [c.line for c in verifier.verify(p, cands, ev, runs=1)["kept"]] == [9]


def test_repeated_tool_finding_in_a_file_is_listed_once():
    p = pr()
    ev = [
        Evidence("E1", "app/search.py", 12, "rules", "cor.float_money", "correctness", "low", "float for money"),
        Evidence("E2", "app/search.py", 16, "rules", "cor.float_money", "correctness", "low", "float for money"),
    ]
    cands = specialists.tool_comments("correctness", p, ev, {})
    res = verifier.verify(p, cands, ev, runs=1)
    assert [c.line for c in res["kept"]] == [12]
    assert "also at line 16" in res["kept"][0].body and res["kept"][0].evidence == ["E1", "E2"]
    assert any("listed there" in c.drop_reason for c in res["dropped"])


def test_quote_is_located_on_the_closest_matching_line():
    two = DIFF.replace("+    return rows\n", "+    return rows\n+\n+\n+def other():\n+    return rows\n")
    p = PRInput.from_diff(two.replace("@@ -0,0 +1,16 @@", "@@ -0,0 +1,20 @@"))
    c = Comment("app/search.py", 25, "correctness", "low", "x", "y", "correctness/llm", quote="return rows")
    assert verifier._anchor(c, p) is None and c.line == 20


def test_llm_intent_tolerates_prefixed_paths_and_a_single_path_string(monkeypatch):
    def reply(system, user, sample):
        return {
            "requirements": [{"text": "search users", "status": "covered", "files": "b/app/search.py"}],
            "unrequested": [{"file": "b/app/search.py", "reason": "x"}, {"file": "nope.py", "reason": "y"}],
        }

    fake_llm(monkeypatch, reply)
    res = intent.check_intent(pr(issue_title="Search users"), use_llm=True)
    assert res["mode"] == "llm"
    assert res["requirements"][0]["files"] == ["app/search.py"]
    assert [u["file"] for u in res["unrequested"]] == ["app/search.py"]