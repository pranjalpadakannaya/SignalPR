import { Grouped, HBar, Lines } from "../charts/index.jsx";
import { Card, Loading, NotReady, PageHead, Tile } from "../components/ui.jsx";
import { f3, num, pct, prettyAgent, useApi } from "../lib.js";

const LR = "logistic regression (numeric features)", SIZE = "change size (lines changed)";

function SliceTable({ rows, keyLabel, fmtKey = (k) => k }) {
  return (
    <table className="data"><thead><tr><th>{keyLabel}</th><th className="num">Files</th><th className="num">Base rate</th><th className="num">PR-AUC model</th><th className="num">PR-AUC size</th></tr></thead>
      <tbody>{rows.map((r) => <tr key={r.key}><td>{fmtKey(r.key)}</td><td className="num">{num(r.n)}</td><td className="num">{pct(r.base_rate, 2)}</td>
        <td className="num"><b>{f3(r.pr_auc)}</b></td><td className="num">{f3(r.size_pr_auc)}</td></tr>)}</tbody></table>
  );
}

export default function Model() {
  const { data: m, error, loading } = useApi("/api/model");
  if (loading) return <Loading />;
  if (error) return <NotReady title="No trained model yet" cmd="python -m risk train && python -m risk report" />;
  const T = m.test, B = m.baselines, R = m.ranking, cal = T.calibration;
  const cmax = Math.max(...cal.map((c) => Math.max(c.pred, c.obs))) * 1.08;
  const prSeries = [{ name: "Risk model", color: "var(--series-1)", points: T.pr_curve }];
  if (B[LR]?.pr_curve) prSeries.push({ name: "Logistic regression", color: "var(--series-2)", points: B[LR].pr_curve });
  if (B[SIZE]?.pr_curve) prSeries.push({ name: "Change size", color: "var(--baseline)", points: B[SIZE].pr_curve, dash: true });
  const keys = ["hit@1", "recall@3", "recall@5"];
  return (
    <>
      <PageHead title="Risk model">
        Gradient-boosted trees ({m.backend}) predicting whether a changed file draws a human inline review comment, from {m.n_features} features
        available before review. Tested on <b>{num(m.data.test_repos)} repositories the model never saw, using PRs newer than any training PR</b>.
      </PageHead>
      <div className="grid g4">
        <Tile label="PR-AUC (test)" value={f3(T.pr_auc)} sub={`95% CI ${f3(T.pr_auc_ci95[0])}–${f3(T.pr_auc_ci95[1])}`} />
        <Tile label="Lift over base rate" value={`${(T.pr_auc / T.base_rate).toFixed(1)}×`} sub={`base rate ${pct(T.base_rate, 2)}`} />
        <Tile label="ROC-AUC (test)" value={f3(T.roc_auc)} sub={`${num(T.n)} files, ${num(T.positives)} positives`} />
        <Tile label="Calibration error (ECE)" value={f3(T.ece)} sub={`Brier ${f3(T.brier)}`} />
      </div>
      <div className="grid g2 mt">
        <Card title="PR-AUC vs baselines" sub="Same test set. Whiskers: 95% bootstrap CI (resampling pull requests).">
          <HBar rows={[{ label: "SignalPR risk model", value: T.pr_auc, ci: T.pr_auc_ci95, hl: true },
            ...Object.entries(B).map(([k, v]) => ({ label: k.replace(" (numeric features)", "").replace(" (lines changed)", ""), value: v.pr_auc, ci: v.pr_auc_ci95, color: "var(--baseline)" }))]}
            fmt={(v) => v.toFixed(3)} labelW={190} valueW={60} valueLabel="PR-AUC" />
        </Card>
        <Card title="Precision–recall curve" sub="Dashed horizontal line: base rate (a random ranker).">
          <Lines series={prSeries} xMax={1} yMax={1} xLabel="Recall" yLabel="Precision" xFmt={(v) => v.toFixed(2)} yFmt={(v) => v.toFixed(2)} hline={T.base_rate} />
        </Card>
      </div>
      <div className="grid g2 mt">
        <Card title="Ranking files inside a PR" sub={`What the agent's ranker relies on: on ${num(R["risk model"].n_prs)} test PRs with ≥ 2 files and ≥ 1 human comment, how often the commented files come first.`}>
          <Grouped groups={["Top file commented", "Recall in top 3", "Recall in top 5"]} fmt={(v) => pct(v, 0)} max={1} series={[
            { name: "Risk model", color: "var(--series-1)", values: keys.map((k) => R["risk model"][k]) },
            { name: "Change size", color: "var(--series-2)", values: keys.map((k) => R["change size"][k]) },
            { name: "Random", color: "var(--baseline)", values: keys.map((k) => R.random[k]) }]} />
        </Card>
        <Card title="Calibration" sub="Predicted vs observed comment rate (10 quantile bins). On the dashed line = calibrated.">
          <Lines series={[{ name: "Model", color: "var(--series-1)", points: cal.map((c) => [c.pred, c.obs]), markers: true }]}
            xMax={cmax} yMax={cmax} diagonal xLabel="Predicted rate" yLabel="Observed rate" xFmt={(v) => pct(v, 1)} yFmt={(v) => pct(v, 1)} />
        </Card>
      </div>
      <div className="grid g2 mt">
        <Card title="Which features predict reviewer attention" sub="Permutation importance: drop in PR-AUC when the feature is shuffled.">
          <HBar rows={m.feature_importance.slice(0, 15).map((r) => ({ label: r.feature, value: Math.max(0, r.importance) }))} fmt={(v) => v.toFixed(3)} valueLabel="PR-AUC drop" />
        </Card>
        <Card title="By agent (test)">
          <SliceTable rows={m.by_agent} keyLabel="Agent" fmtKey={prettyAgent} />
          <h3 className="mt">By language (test)</h3>
          <SliceTable rows={m.by_lang.slice(0, 8)} keyLabel="Language" />
          <h3 className="mt">Data split</h3>
          <table className="data"><tbody>
            <tr><td>Train</td><td className="num">{num(m.data.train_files)} files · {num(m.data.train_repos)} repos</td><td className="num">{pct(m.data.train_pos_rate, 2)} positive</td></tr>
            <tr><td>Test (unseen repos, newer PRs)</td><td className="num">{num(m.data.test_files)} files · {num(m.data.test_repos)} repos</td><td className="num">{pct(m.data.test_pos_rate, 2)} positive</td></tr>
            <tr><td>Unseen repos, all dates</td><td className="num">PR-AUC {f3(m.test_unseen_repos_all_time.pr_auc)}</td><td className="num">ROC-AUC {f3(m.test_unseen_repos_all_time.roc_auc)}</td></tr>
          </tbody></table>
        </Card>
      </div>
    </>
  );
}
