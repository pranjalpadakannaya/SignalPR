import math

import pandas as pd

from risk.features import FEATURES, PATCH_FEATURES, build_features, patch_features


def frame(**pr):
    rows = [
        {"pr_id": 1, "filename": "src/api/user_search.py", "status": "added", "additions": 40, "deletions": 0},
        {"pr_id": 1, "filename": "tests/test_user_search.py", "status": "added", "additions": 20, "deletions": 0},
        {"pr_id": 1, "filename": "package-lock.json", "status": "modified", "additions": 900, "deletions": 800},
    ]
    return pd.DataFrame([{**r, **pr} for r in rows])


def test_all_features_present_and_path_semantics():
    f = build_features(frame(agent="Copilot", title="Add user search"))
    assert set(FEATURES) <= set(f.columns)
    assert f.loc[0, "tok_api"] == 1 and f.loc[1, "is_test"] == 1 and f.loc[2, "is_lock"] == 1
    assert f.loc[0, "pr_n_files"] == 3 and f.loc[0, "pr_has_tests"] == 1
    assert f.loc[0, "agent"] == "Copilot" and f.loc[0, "task_type"] == "feat"


def test_issue_alignment_uses_issue_text_then_falls_back_to_title():
    f = build_features(frame(title="Refactor", issue_text="Users need search by name"))
    assert f.loc[0, "issue_path_hits"] >= 2  # user, search
    assert f.loc[2, "issue_path_hits"] == 0  # lockfile unrelated
    g = build_features(frame(title="Add user search endpoint"))  # no issue -> PR title used
    assert g.loc[0, "issue_path_hits"] >= 2 and math.isnan(g.loc[0, "pr_issue_overlap"])


def test_patch_features():
    patch = (
        "@@ -1,2 +1,6 @@\n x = 1\n-if a:\n+if a and b:\n+    eval(x)\n"
        "+    total = total + price * qty\n+    total = total + price * qty\n"
    )
    pf = patch_features(patch, "a.py")
    assert pf["complexity_delta"] == 1  # if/and added vs if removed
    assert pf["repeated_added_lines"] == 2  # duplicated line
    assert pf["tool_security"] == 1 and pf["n_hunks"] == 1
    assert all(math.isnan(v) for v in patch_features(None, "a.py").values())
    f = build_features(frame().assign(patch=[patch, None, None]))
    assert f.loc[0, "tool_warnings"] >= 1 and math.isnan(f.loc[1, "tool_warnings"])
    assert set(PATCH_FEATURES) <= set(f.columns)
