from benchmarks import martian, reviewbench
from benchmarks.authorship import classify
from benchmarks.common import CLOSING_RE, files_from_diff, parse_pr_url
from benchmarks.stats import compare, summarize, validate
from tests.benchmarks.test_schema_registry import make_example

DIFF = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1,2 +1,3 @@
 x = 1
-y = 2
+y = 3
+z = 4
diff --git a/new.txt b/new.txt
new file mode 100644
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+hello
diff --git a/old.py b/moved.py
similarity index 90%
rename from old.py
rename to moved.py
"""


def test_files_from_diff():
    files = {f.path: f for f in files_from_diff(DIFF)}
    assert files["app.py"].additions == 2 and files["app.py"].deletions == 1
    assert files["new.txt"].status == "added"
    assert files["moved.py"].status == "renamed" and files["moved.py"].old_path == "old.py"


def test_parse_pr_url():
    assert parse_pr_url("https://github.com/getsentry/sentry/pull/93824") == ("getsentry/sentry", 93824)
    assert parse_pr_url("https://github.com/discourse/discourse/commit/abc") is None


def test_closing_refs():
    body = "Fixes #12. Also closes https://github.com/o/r/issues/34 and resolves other/repo#5"
    refs = [(m.group(1), int(m.group(2))) for m in CLOSING_RE.finditer(body)]
    assert refs == [(None, 12), ("o/r", 34), ("other/repo", 5)]


def test_martian_known_issues_profiles():
    issues = martian.known_issues(
        "martian/x",
        [
            {"comment": " Null deref ", "severity": "Critical", "category": "bug"},
            {"comment": "Slow loop", "severity": "Low", "category": "perf"},
            {"comment": "Naming", "severity": "Low", "category": "style"},
        ],
    )
    assert issues[0].body == "Null deref" and issues[0].severity == "critical" and issues[0].file is None
    assert issues[0].extra["profiles"] == ["strict", "core", "all"]
    assert issues[1].extra["profiles"] == ["core", "all"]
    assert issues[2].extra["profiles"] == ["all"]


def test_reviewbench_key_and_findings():
    entry = {
        "nwo": "Owner/repo",
        "repo": "https://github.com/Owner/repo",
        "pr_number": 17,
        "head": "a1978cb7" + "0" * 32,
    }
    assert reviewbench.pr_key(entry) == "Owner_repo_17-a1978cb7"
    assert reviewbench.pr_key({k: v for k, v in entry.items() if k != "nwo"}) == "Owner_repo_17-a1978cb7"
    issues = reviewbench.known_issues(
        "reviewbench/Owner_repo_17",
        [
            {
                "file": "a.py",
                "start_line": 3,
                "end_line": 5,
                "message": "bug",
                "tp_fp": "fp",
                "severity": "medium",
                "category": "correctness",
                "difficulty": "easy",
                "source": {"type": "llm_review"},
            }
        ],
    )
    k = issues[0]
    assert (k.file, k.start_line, k.end_line, k.label, k.source) == ("a.py", 3, 5, "fp", "llm_review")
    assert k.extra == {"difficulty": "easy"}


def test_authorship():
    assert classify(None)["author_kind"] == "unknown"
    assert classify({"author": {"login": "Copilot", "__typename": "Bot"}})["agent"] == "Copilot"
    assert classify({"author": {"login": "alice"}, "headRefName": "codex/fix-bug"})["agent"] == "OpenAI_Codex"
    claude = classify({"author": {"login": "bob"}, "body": "Summary\n\nGenerated with [Claude Code](https://x)"})
    assert claude["agent"] == "Claude_Code" and claude["agent_signal"] == "body"
    trailer = classify(
        {
            "author": {"login": "bob"},
            "commit_messages": ["fix\n\nCo-authored-by: Cursor Agent <cursoragent@cursor.com>"],
        }
    )
    assert trailer["agent"] == "Cursor" and trailer["agent_signal"] == "commits"
    assert classify({"author": {"login": "dependabot[bot]", "__typename": "Bot"}})["author_kind"] == "bot"
    human = classify({"author": {"login": "carol"}, "headRefName": "feature/x", "body": "Adds a thing"})
    assert human["author_kind"] == "human" and human["agent"] is None


def test_summarize_and_validate():
    exs = [
        make_example(metadata={"language": "Python", "size_bucket": "xs", "author_kind": "human"}),
        make_example(id="martian/o_r_2", metadata={"language": "Go", "size_bucket": "s", "author_kind": "agent"}),
    ]
    s = summarize(exs)
    assert s["prs"] == 2 and s["findings"] == 4 and s["true_positives"] == 2
    assert s["by_severity_tp"] == {"high": 2} and s["by_language"] == {"Python": 1, "Go": 1}
    rows = validate("reviewbench", exs, {"prs": 2, "by_language": {"Python": 1, "Rust": 1}})
    assert [(r[0], r[3]) for r in rows] == [("prs", True), ("by_language.Python", True), ("by_language.Rust", False)]


def test_compare_nested_missing():
    assert compare({"a": {"b": 1}}, {}) == [("a.b", 1, None, False)]
