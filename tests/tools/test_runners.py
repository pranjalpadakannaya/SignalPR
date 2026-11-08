import shutil

import pytest

from tools import files_from_diff, run_tools

DIFF = """diff --git a/app/db.py b/app/db.py
new file mode 100644
--- /dev/null
+++ b/app/db.py
@@ -0,0 +1,8 @@
+import os
+API_KEY = "not-a-real-key-0123456789"
+def find(conn, name, cache={}):
+    try:
+        return conn.execute(f"SELECT * FROM users WHERE name = '{name}'").fetchall()
+    except:
+        print("failed")
+        return []
diff --git a/web/cart.js b/web/cart.js
new file mode 100644
--- /dev/null
+++ b/web/cart.js
@@ -0,0 +1,3 @@
+export function show(el, coupon) {
+  if (coupon == "SAVE10") { el.innerHTML = "<p>" + coupon + "</p>"; }
+}
"""


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.runners.CACHE_DIR", tmp_path / "cache")


def test_builtin_rules_find_planted_issues():
    rep = run_tools(files_from_diff(DIFF), tools=("rules",))
    rules = {(f.file, f.rule) for f in rep.findings}
    assert ("app/db.py", "sec.secret") in rules
    assert ("app/db.py", "sec.sql_fstring") in rules or ("app/db.py", "sec.sql_concat") in rules
    assert ("app/db.py", "cor.bare_except") in rules
    assert ("app/db.py", "cor.mutable_default") in rules
    assert ("web/cart.js", "sec.innerhtml") in rules
    assert all(f.line in range(1, 9) for f in rep.findings)


def test_findings_only_on_changed_lines():
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 x = eval(input())
+y = 1
 z = 2
 w = 3
"""
    rep = run_tools(files_from_diff(diff), tools=("rules",))
    assert rep.findings == []  # the eval is on an unchanged context line


@pytest.mark.skipif(not shutil.which("ruff"), reason="ruff not installed")
def test_ruff_runs_and_cache_hits():
    files = files_from_diff(DIFF)
    first = run_tools(files, tools=("ruff",))
    assert any(f.tool == "ruff" for f in first.findings)
    second = run_tools(files, tools=("ruff",))
    assert second.cache_hits >= 1
    assert [f.to_dict() for f in second.findings] == [f.to_dict() for f in first.findings]


def test_counts_for_features():
    rep = run_tools(files_from_diff(DIFF), tools=("rules",))
    c = rep.counts("app/db.py")
    assert c["tool_warnings"] >= 4 and c["tool_security"] >= 2 and c["tool_critical"] >= 2


def test_missing_tool_reported_not_crashing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    rep = run_tools(files_from_diff(DIFF), tools=("ruff", "eslint", "lizard"))
    assert rep.status["ruff"] == "not installed"
