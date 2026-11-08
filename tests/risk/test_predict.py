import pytest

from risk import predict_risk
from risk.paths import MODEL_PATH
from tools.diff import parse_unified_diff

FILES = [
    {"filename": "app/routes/search.py", "status": "added", "additions": 30, "deletions": 0},
    {"filename": "README.md", "status": "modified", "additions": 2, "deletions": 1},
    {"filename": "package-lock.json", "status": "modified", "additions": 300, "deletions": 250},
]


def test_shape_and_order():
    out = predict_risk(FILES, {"agent": "OpenAI_Codex", "title": "Add user search endpoint", "stars": 640})
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert all(0 <= r["risk"] <= 1 for r in out) and out[0]["risk_rel"] == pytest.approx(1.0)
    assert out[0]["risk"] >= out[-1]["risk"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model")
def test_trained_model_prefers_code_over_lockfile():
    out = {r["file"]: r["risk"] for r in predict_risk(FILES, {"agent": "Copilot", "title": "Add user search"})}
    assert out["app/routes/search.py"] > out["package-lock.json"]
    assert "risk model" in predict_risk(FILES)[0]["source"]


def test_accepts_filediff_and_unknown_agent():
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n x = 1\n+y = eval(x)\n"
    out = predict_risk(parse_unified_diff(diff), {"agent": "SomeHuman"})
    assert out[0]["file"] == "a.py" and out[0]["additions"] == 1


def test_empty():
    assert predict_risk([]) == []


def test_model_file_roundtrip(tmp_path):
    import numpy as np
    import pandas as pd

    from risk.features import build_features
    from risk.model import RiskModel

    m = RiskModel.load()
    if m is None:
        pytest.skip("no trained model")
    p = tmp_path / "m.joblib"
    m.save(p)
    m2 = RiskModel.load(p)
    feats = build_features(pd.DataFrame(FILES).assign(pr_id=0))
    assert np.allclose(m.predict(feats), m2.predict(feats))
    assert m2.features == m.features and m2.calib_x is not None


def test_missing_or_bad_model_file(tmp_path):
    from risk.model import RiskModel

    assert RiskModel.load(tmp_path / "nope.joblib") is None
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.warns(UserWarning):
        assert RiskModel.load(bad) is None
