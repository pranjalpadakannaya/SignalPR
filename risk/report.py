"""Figures + summary tables for the progress/final report (KDD template friendly: PDF + PNG).

python -m risk report      → results/risk/figures/*.{pdf,png}, results/risk/summary.md
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from risk.paths import FIG_DIR, INSIGHTS_PATH, METRICS_PATH, RESULTS_DIR  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a1", "#52514e"
AGENT_COLORS = {
    "Claude_Code": "#2a78d6",
    "Copilot": "#eb6834",
    "OpenAI_Codex": "#1baf7a",
    "Cursor": "#eda100",
    "Devin": "#e87ba4",
    "Google_Jules": "#008300",
}
PRETTY = {"Claude_Code": "Claude Code", "OpenAI_Codex": "OpenAI Codex", "Google_Jules": "Google Jules"}


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cfcdc6",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": "#ecebe7",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    return name


def _hbar(ax, labels, values, colors, fmt="{:.1%}", ci=None):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62)
    if ci:
        for yi, (lo, hi) in zip(y, ci):
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2)
    ax.set_yticks(list(y), labels)
    ax.grid(axis="y", visible=False)
    for yi, v, c in zip(y, values, ci or [None] * len(values)):
        ax.text((c[1] if c else v) + max(values) * 0.015, yi, fmt.format(v), va="center", fontsize=8, color="#0b0b0b")
    ax.set_xlim(0, max([*(c[1] for c in ci)] if ci else values) * 1.18)
    if "%" in fmt:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))


def make_figures() -> dict:
    _style()
    m = json.loads(METRICS_PATH.read_text())
    ins = json.loads(INSIGHTS_PATH.read_text()) if INSIGHTS_PATH.exists() else None
    T, B = m["test"], m["baselines"]
    made = []

    # 1. PR-AUC vs baselines with CIs
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    names = ["Risk model (ours)"] + [k.split(" (")[0].capitalize() for k in B]
    vals = [T["pr_auc"]] + [v["pr_auc"] for v in B.values()]
    cis = [T["pr_auc_ci95"]] + [v["pr_auc_ci95"] for v in B.values()]
    _hbar(ax, names, vals, [BLUE] + [GRAY] * len(B), fmt="{:.3f}", ci=cis)
    ax.set_xlabel("PR-AUC on unseen repos (95% CI)")
    made.append(_save(fig, "pr_auc_baselines"))

    # 2. precision-recall curves
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(*zip(*T["pr_curve"]), color=BLUE, lw=2, label=f"Risk model (AP {T['pr_auc']:.3f})")
    styles = {"logistic regression (numeric features)": (ORANGE, "-"), "change size (lines changed)": (GRAY, "--")}
    for k, (c, ls) in styles.items():
        if k in B and B[k].get("pr_curve"):
            ax.plot(
                *zip(*B[k]["pr_curve"]),
                color=c,
                lw=1.6,
                ls=ls,
                label=f"{k.split(' (')[0].capitalize()} (AP {B[k]['pr_auc']:.3f})",
            )
    ax.axhline(T["base_rate"], color=GRAY, lw=1, ls=":", label=f"Random ({T['base_rate']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7, loc="upper right")
    made.append(_save(fig, "pr_curve"))

    # 3. calibration
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    cal = T["calibration"]
    mx = max(max(c["pred"], c["obs"]) for c in cal) * 1.08
    ax.plot([0, mx], [0, mx], color=GRAY, ls="--", lw=1)
    ax.plot([c["pred"] for c in cal], [c["obs"] for c in cal], color=BLUE, lw=2, marker="o", ms=4)
    ax.set(
        xlabel="Predicted comment rate",
        ylabel="Observed comment rate",
        xlim=(0, mx),
        ylim=(0, mx),
        title=f"ECE = {T['ece']:.4f}",
    )
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    made.append(_save(fig, "calibration"))

    # 4. feature importance
    fi = [r for r in m["feature_importance"] if r["importance"] > 0][:12]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    _hbar(ax, [r["feature"] for r in fi], [r["importance"] for r in fi], [BLUE] * len(fi), fmt="{:.3f}")
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    made.append(_save(fig, "feature_importance"))

    # 5. within-PR ranking
    R = m["ranking"]
    keys, groups = ["hit@1", "recall@3", "recall@5"], ["Top file", "Top 3", "Top 5"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.26
    for i, (name, c) in enumerate([("risk model", BLUE), ("change size", ORANGE), ("random", GRAY)]):
        ax.bar(
            [x + (i - 1) * w for x in range(3)], [R[name][k] for k in keys], width=w, color=c, label=name.capitalize()
        )
    ax.set_xticks(range(3), groups)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_ylabel("Commented files found")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    made.append(_save(fig, "within_pr_ranking"))

    if ins:
        # 6. comment rate vs change size
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        rc = ins["rate_by_churn"]
        ax.bar([r["key"] for r in rc], [r["rate"] for r in rc], color=BLUE, width=0.62)
        ax.set(xlabel="Lines changed in file", ylabel="Files commented on")
        ax.tick_params(axis="x", labelrotation=35, labelsize=7.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
        made.append(_save(fig, "rate_by_change_size"))
        # 7. comment rate by file category
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        cat = ins["rate_by_category"]
        _hbar(ax, [r["key"] for r in cat], [r["rate"] for r in cat], [BLUE] * len(cat))
        ax.set_xlabel("Files commented on")
        made.append(_save(fig, "rate_by_file_category"))
        # 8. agent merge rate
        ag = ins["agents"]
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        _hbar(
            ax,
            [PRETTY.get(a["agent"], a["agent"]) for a in ag],
            [a["merge_rate"] for a in ag],
            [AGENT_COLORS.get(a["agent"], GRAY) for a in ag],
            fmt="{:.0%}",
        )
        ax.set_xlabel("PRs merged")
        made.append(_save(fig, "merge_rate_by_agent"))
    return {"figures": made, "summary": write_summary(m, ins)}


def write_summary(m: dict, ins: dict | None) -> str:
    T, B, R, D = m["test"], m["baselines"], m["ranking"], m["data"]
    lines = ["# Risk model results (auto-generated by `python -m risk report`)", ""]
    if ins:
        t = ins["totals"]
        lines += [
            "## Data",
            "",
            f"- Source: {ins['source']}",
            f"- {t['prs']:,} agent PRs in {t['repos']:,} repos ({t['date_min'][:10]} → {t['date_max'][:10]}); "
            f"{t['file_changes']:,} changed-file rows",
            f"- {t['inline_comments_human']:,} human inline comments "
            f"({t['inline_comments_bot']:,} bot comments dropped)",
            f"- Population: {t['population_prs']:,} merged, human-reviewed PRs → {t['population_files']:,} files; "
            f"{t['positive_files']:,} commented ({t['positive_rate']:.2%})",
            f"- Patch features available: {ins.get('patch_features', False)}",
            "",
        ]
    lines += [
        "## Split",
        "",
        "| Split | Files | Repos | Positive rate |",
        "|---|---:|---:|---:|",
        f"| Train | {D['train_files']:,} | {D['train_repos']:,} | {D['train_pos_rate']:.2%} |",
        f"| Test (unseen repos, newer PRs) | {D['test_files']:,} | {D['test_repos']:,} | {D['test_pos_rate']:.2%} |",
        "",
        f"## Test results ({m['backend']}, {m['n_features']} features)",
        "",
        "| Ranker | PR-AUC (95% CI) | ROC-AUC |",
        "|---|---|---:|",
        f"| **Risk model** | **{T['pr_auc']:.3f}** ({T['pr_auc_ci95'][0]:.3f}–{T['pr_auc_ci95'][1]:.3f}) "
        f"| **{T['roc_auc']:.3f}** |",
    ]
    for k, v in B.items():
        lines.append(
            f"| {k} | {v['pr_auc']:.3f} ({v['pr_auc_ci95'][0]:.3f}–{v['pr_auc_ci95'][1]:.3f}) | {v['roc_auc']:.3f} |"
        )
    lines += [
        "",
        f"Base rate {T['base_rate']:.4f} · lift {T['pr_auc'] / T['base_rate']:.1f}× · "
        f"ECE {T['ece']:.4f} · Brier {T['brier']:.4f} · unseen repos, all dates: PR-AUC "
        f"{m['test_unseen_repos_all_time']['pr_auc']:.3f}",
        "",
        f"## Within-PR ranking ({R['risk model']['n_prs']} test PRs with ≥2 files and ≥1 comment)",
        "",
        "| | Top file commented | Recall@3 | Recall@5 |",
        "|---|---:|---:|---:|",
    ]
    for k in ("risk model", "change size", "random"):
        lines.append(f"| {k} | {R[k]['hit@1']:.1%} | {R[k]['recall@3']:.1%} | {R[k]['recall@5']:.1%} |")
    lines += ["", "## Top features (permutation importance, PR-AUC drop)", ""]
    lines += [f"{i}. `{r['feature']}`: {r['importance']:.4f}" for i, r in enumerate(m["feature_importance"][:12], 1)]
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary.md").write_text(text)
    return str(RESULTS_DIR / "summary.md")
