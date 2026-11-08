"""Risk model: predicts which changed files human reviewers comment on.

Uses LightGBM when installed (as in the proposal) and falls back to scikit-learn's
HistGradientBoostingClassifier (same family: histogram gradient-boosted trees) otherwise.
Probabilities are calibrated with isotonic regression on the validation split. (issue #10, task 5)
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from risk.features import CATEGORICAL, FEATURES, NUMERIC
from risk.paths import METRICS_PATH, MODEL_PATH, PROCESSED_DIR

try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:  # pragma: no cover
    HAS_LGB = False


FORMAT = "signalpr-risk-v1"


def _arr(v):
    return None if v is None else np.asarray(v, dtype=float)


@dataclass
class RiskModel:
    """Trained model + everything needed to score new files.

    Saved in a version-independent way: categorical maps and isotonic calibration as plain
    lists, and a LightGBM model as its text dump, so the file loads with any scikit-learn version.
    (The scikit-learn fallback model is pickled and needs the same scikit-learn version to load.)
    """

    backend: str
    model: object
    cat_maps: dict[str, dict[str, int]]
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    meta: dict = field(default_factory=dict)
    calib_x: np.ndarray | None = None  # isotonic calibration: raw score -> probability (piecewise linear)
    calib_y: np.ndarray | None = None

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in self.features:
            if c in CATEGORICAL:
                m = self.cat_maps[c]
                X[c] = df[c].astype(str).map(m).fillna(m.get("__other__", 0)).astype(np.int32)
            else:
                X[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        return X

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        X = self.encode(df)
        if self.backend == "lightgbm":
            return self.model.predict(X, num_iteration=getattr(self.model, "best_iteration", None) or None)
        return self.model.predict_proba(X)[:, 1]

    def calibrate(self, raw: np.ndarray, y: np.ndarray) -> None:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y)
        self.calib_x = np.asarray(iso.X_thresholds_, dtype=float)
        self.calib_y = np.asarray(iso.y_thresholds_, dtype=float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        s = self.raw_score(df)
        if self.calib_x is None:
            return s
        return np.interp(s, self.calib_x, self.calib_y)

    def save(self, path=None):
        path = Path(path or MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model.model_to_string() if self.backend == "lightgbm" else self.model
        joblib.dump(
            {
                "format": FORMAT,
                "backend": self.backend,
                "model": model,
                "cat_maps": self.cat_maps,
                "features": list(self.features),
                "meta": self.meta,
                "calib_x": None if self.calib_x is None else self.calib_x.tolist(),
                "calib_y": None if self.calib_y is None else self.calib_y.tolist(),
            },
            path,
        )

    @staticmethod
    def load(path=None) -> "RiskModel | None":
        path = Path(path or MODEL_PATH)
        if not path.exists():
            return None
        try:
            d = joblib.load(path)
            if d.get("format") != FORMAT:
                raise ValueError(f"unknown model format {d.get('format')!r}")
            model = d["model"]
            if d["backend"] == "lightgbm":
                import lightgbm

                model = lightgbm.Booster(model_str=model)
            return RiskModel(
                d["backend"], model, d["cat_maps"], d["features"], d["meta"], _arr(d["calib_x"]), _arr(d["calib_y"])
            )
        except Exception as e:  # noqa: BLE001 - report and let callers fall back
            warnings.warn(
                f"Could not load risk model {path}: {type(e).__name__}: {e}. Retrain with "
                "`python -m risk train` or get the model file from the team.",
                stacklevel=2,
            )
            return None


def _cat_maps(train: pd.DataFrame, min_count: int = 30) -> dict[str, dict[str, int]]:
    maps = {}
    for c in CATEGORICAL:
        vc = train[c].astype(str).value_counts()
        keep = vc[vc >= min_count].index.tolist()
        maps[c] = {"__other__": 0, **{v: i + 1 for i, v in enumerate(keep)}}
    return maps


def _fit(backend, Xtr, ytr, Xva, yva, cat_cols, seed=0):
    if backend == "lightgbm":
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=100,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            metric="average_precision",
            verbose=-1,
            seed=seed,
            num_threads=0,
        )
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, categorical_feature=cat_cols, reference=dtr)
        return lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    mask = [c in cat_cols for c in Xtr.columns]
    m = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=600,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features=mask,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        scoring="average_precision",
        random_state=seed,
    )
    m.fit(Xtr, ytr)
    return m


# ------------------------------------------------------------------ metrics
def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    out, e = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"pred": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e), out


def _curve_points(x, y, n=120):
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [[float(x[i]), float(y[i])] for i in idx]


def cluster_bootstrap_ap(y, s, groups, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "s": s, "g": groups})
    gs = df["g"].unique()
    by = {g: idx for g, idx in df.groupby("g").indices.items()}
    vals = []
    for _ in range(reps):
        pick = rng.choice(gs, size=len(gs), replace=True)
        ii = np.concatenate([by[g] for g in pick])
        yy = df["y"].values[ii]
        if yy.sum() == 0:
            continue
        vals.append(average_precision_score(yy, df["s"].values[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ranking_metrics(df: pd.DataFrame, score_col: str, ks=(1, 3, 5)) -> dict:
    """Within-PR file ranking on PRs that have >=1 commented file and >=2 files."""
    d = df[(df["pr_has_inline"] == 1) & (df["pr_n_files"] >= 2)]
    res = {f"recall@{k}": [] for k in ks}
    res["hit@1"] = []
    for _, g in d.groupby("pr_id", sort=False):
        order = g.sort_values(score_col, ascending=False)["label"].values
        pos = order.sum()
        if pos == 0:
            continue
        for k in ks:
            res[f"recall@{k}"].append(order[:k].sum() / pos)
        res["hit@1"].append(order[0])
    out = {k: float(np.mean(v)) for k, v in res.items() if v}
    out["n_prs"] = int(len(res["hit@1"]))
    return out


def evaluate(y, p, groups) -> dict:
    prec, rec, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    e, cal = ece(y, p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_ci95": cluster_bootstrap_ap(y, p, groups),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "ece": e,
        "calibration": cal,
        "base_rate": float(np.mean(y)),
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "pr_curve": _curve_points(rec[::-1], prec[::-1]),
        "roc_curve": _curve_points(fpr, tpr),
    }


# ------------------------------------------------------------------ training
def train(sample_frac: float = 1.0, seed: int = 0, verbose: bool = True) -> dict:
    t0 = time.time()
    log = (lambda m: print(f"[train {time.time() - t0:6.1f}s] {m}", flush=True)) if verbose else (lambda m: None)
    df = pd.read_pickle(PROCESSED_DIR / "files.pkl")
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    if sample_frac < 1:
        tr = tr.sample(frac=sample_frac, random_state=seed)
    te_repo = df[df["repo_split"] == "test"]  # unseen repos, any time
    log(f"train={len(tr):,} val={len(va):,} test={len(te):,} (test-repos all-time={len(te_repo):,})")

    backend = "lightgbm" if HAS_LGB else "sklearn-hgb"
    maps = _cat_maps(tr)
    active = [f for f in FEATURES if f in CATEGORICAL or tr[f].notna().any()]
    skipped = [f for f in FEATURES if f not in active]
    if skipped:
        log(f"skipping {len(skipped)} feature(s) with no data in train: {', '.join(skipped)}")
    rm = RiskModel(backend, None, maps, features=active)
    Xtr, Xva = rm.encode(tr), rm.encode(va)
    rm.model = _fit(backend, Xtr, tr["label"].values, Xva, va["label"].values, CATEGORICAL, seed)
    log(f"fitted {backend}")
    rm.calibrate(rm.raw_score(va), va["label"].values)
    log("calibrated on val")

    p_te = rm.predict(te)
    metrics = {"backend": backend, "features": active, "n_features": len(active), "skipped_features": skipped}
    metrics["test"] = evaluate(te["label"].values, p_te, te["pr_id"].values)
    p_tr = rm.predict(te_repo)
    metrics["test_unseen_repos_all_time"] = {
        k: v
        for k, v in evaluate(te_repo["label"].values, p_tr, te_repo["pr_id"].values).items()
        if k not in ("pr_curve", "roc_curve", "calibration")
    }
    log(f"test PR-AUC={metrics['test']['pr_auc']:.4f} ROC-AUC={metrics['test']['roc_auc']:.4f}")

    # ---- baselines ----
    base = {}
    ytest, g = te["label"].values, te["pr_id"].values
    base["change size (lines changed)"] = te["changes"].values
    rng = np.random.default_rng(seed)
    base["random"] = rng.random(len(te))
    lr_cols = [c for c in NUMERIC if c in active]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.5))
    lr_tr = tr.sample(n=min(len(tr), 300_000), random_state=seed)
    lr.fit(lr_tr[lr_cols].fillna(0).values, lr_tr["label"].values)
    base["logistic regression (numeric features)"] = lr.predict_proba(te[lr_cols].fillna(0).values)[:, 1]
    metrics["baselines"] = {}
    for name, s in base.items():
        metrics["baselines"][name] = {
            "pr_auc": float(average_precision_score(ytest, s)),
            "pr_auc_ci95": cluster_bootstrap_ap(ytest, s, g, reps=100),
            "roc_auc": float(roc_auc_score(ytest, s)),
        }
        prec, rec, _ = precision_recall_curve(ytest, s)
        metrics["baselines"][name]["pr_curve"] = _curve_points(rec[::-1], prec[::-1], n=60)
    log("baselines done")

    # ---- within-PR ranking (what the agent's ranker relies on) ----
    tt = te.assign(_model=p_te, _size=te["changes"].values, _rand=base["random"])
    metrics["ranking"] = {
        "risk model": ranking_metrics(tt, "_model"),
        "change size": ranking_metrics(tt, "_size"),
        "random": ranking_metrics(tt, "_rand"),
    }

    # ---- per-agent and per-language PR-AUC on test ----
    for key in ("agent", "lang"):
        rows = []
        for v, sub in tt.groupby(key):
            if sub["label"].sum() >= 15:
                rows.append(
                    {
                        "key": v,
                        "pr_auc": float(average_precision_score(sub["label"], sub["_model"])),
                        "size_pr_auc": float(average_precision_score(sub["label"], sub["_size"])),
                        "base_rate": float(sub["label"].mean()),
                        "n": int(len(sub)),
                    }
                )
        metrics[f"by_{key}"] = sorted(rows, key=lambda r: -r["n"])[:14]

    # ---- feature importance: permutation on a test subsample (PR-AUC drop) ----
    metrics["feature_importance"] = permutation_importance(rm, te, seed=seed)
    if backend == "lightgbm":
        gain = rm.model.feature_importance("gain")
        metrics["gain_importance"] = sorted(
            [{"feature": f, "gain": float(v)} for f, v in zip(Xtr.columns, gain)], key=lambda r: -r["gain"]
        )[:25]
    log("feature importance done")

    metrics["data"] = {
        "train_files": int(len(tr)),
        "val_files": int(len(va)),
        "test_files": int(len(te)),
        "train_repos": int(tr["repo_id"].nunique()),
        "test_repos": int(te["repo_id"].nunique()),
        "train_pos_rate": float(tr["label"].mean()),
        "test_pos_rate": float(te["label"].mean()),
    }
    metrics["trained_seconds"] = round(time.time() - t0, 1)
    rm.meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "backend": backend,
        "test_pr_auc": metrics["test"]["pr_auc"],
        "base_rate": metrics["data"]["train_pos_rate"],
    }
    rm.save()
    METRICS_PATH.write_text(json.dumps(metrics, indent=1))
    log(f"saved {MODEL_PATH} and {METRICS_PATH}")
    return metrics


def permutation_importance(rm: RiskModel, te: pd.DataFrame, n: int = 40_000, reps: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = te[te["label"] == 1]
    neg = te[te["label"] == 0]
    sub = pd.concat([pos, neg.sample(n=min(len(neg), n - len(pos)), random_state=seed)])
    y = sub["label"].values
    base = average_precision_score(y, rm.raw_score(sub))
    out = []
    for f in rm.features:
        if sub[f].isna().all():
            continue
        drops = []
        for _ in range(reps):
            s2 = sub.copy()
            s2[f] = rng.permutation(s2[f].values)
            drops.append(base - average_precision_score(y, rm.raw_score(s2)))
        out.append({"feature": f, "importance": float(np.mean(drops))})
    return sorted(out, key=lambda r: -r["importance"])[:25]


if __name__ == "__main__":
    train()
