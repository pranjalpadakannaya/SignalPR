import { useEffect, useRef, useState } from "react";
import { HBar } from "../charts/index.jsx";
import { Callout, Card, Cmd, Loading, PageHead } from "../components/ui.jsx";
import { api, f2, f3, post, useApi } from "../lib.js";
import { useStatus } from "../status.js";

const isFull = (k) => k.startsWith("SignalPR (full");
const Cell = ({ v, ci }) => <>{f3(v)}{ci && <div className="small muted">{f3(ci[0])}–{f3(ci[1])}</div>}</>;

export default function Evaluation() {
  const { status } = useStatus();
  const { data: d, error, loading, reload } = useApi("/api/eval");
  const [run, setRun] = useState(null);
  const timer = useRef();
  const poll = async () => {
    const s = await api("/api/eval/status").catch(() => null);
    setRun(s);
    if (s?.state === "running") timer.current = setTimeout(poll, 2000);
    else if (s?.state === "done") reload();
  };
  useEffect(() => { poll(); return () => clearTimeout(timer.current); }, []);
  const start = async () => { await post("/api/eval/run", { runs: 3 }); poll(); };
  const runBox = (
    <Card title="Run the benchmark" className="mt" sub={`Runs SignalPR, the baselines and every ablation on the demo set ${status?.llm?.available ? `with ${status.llm.model}` : "without an LLM (AI reviewers are off)"}. With an LLM this makes roughly 10 model calls per PR per system, so it can take several minutes on a free tier.`}>
      <div className="row"><button className="btn primary" onClick={start} disabled={run?.state === "running"}>{run?.state === "running" ? "Running…" : "Run evaluation"}</button>
        {run?.state === "running" && <span className="small muted">{run.progress} · {run.elapsed}s</span>}
        {run?.state === "error" && <span className="small" style={{ color: "var(--critical-ink)" }}>{run.error}</span>}</div>
      <p className="small muted mt">Martian (dev) and ReviewBench (test): download them, then run from the terminal:</p>
      <Cmd>{`python -m benchmarks download martian reviewbench
python -m eval --bench martian --repeats 3                   # DEV set
python -m eval --bench reviewbench --subset test_subset      # TEST set, frozen config only`}</Cmd>
    </Card>
  );
  if (loading) return <Loading />;
  if (error) return <><PageHead title="Evaluation" />{runBox}</>;
  const sys = Object.entries(d.systems);
  const demo = d.benchmark === "demo";
  return (
    <>
      <PageHead title="Evaluation">
        Benchmark: <b>{d.benchmark}</b> · {d.n_prs} PRs · {d.n_known_issues} known issues · {d.llm ? `LLM ${d.llm.model}` : "no LLM (analyzer evidence only)"} · {d.created}
      </PageHead>
      {demo && <Callout kind="warn">This is the small built-in <b>demo set</b> (4 PRs with planted issues). It shows the harness works end to end. It is not a research result. For reportable numbers, run ReviewBench / Martian with a real model.</Callout>}
      <Card title="Systems and ablations" className="mt">
        <table className="data"><thead><tr><th>System</th><th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th>
          <th className="num">Severity-weighted recall</th><th className="num">Recall@1</th><th className="num">Recall@3</th><th className="num">Comments per real issue</th><th className="num">Comments</th></tr></thead>
          <tbody>{sys.map(([k, s]) => <tr key={k} className={isFull(k) ? "hl" : ""}><td>{k}</td>
            <td className="num"><Cell v={s.precision} ci={s.ci95?.precision} /></td><td className="num"><Cell v={s.recall} ci={s.ci95?.recall} /></td>
            <td className="num"><Cell v={s.f1} ci={s.ci95?.f1} /></td><td className="num">{f3(s.sev_weighted_recall)}</td>
            <td className="num">{f3(s["recall@1"])}</td><td className="num">{f3(s["recall@3"])}</td>
            <td className="num">{s.comments_per_real_issue ? f2(s.comments_per_real_issue) : "–"}</td><td className="num">{s.comments}</td></tr>)}</tbody></table>
        {Object.keys(d.skipped || {}).length > 0 && <p className="small muted mt-s">Skipped: {Object.entries(d.skipped).map(([k, v]) => `${k} (${v})`).join("; ")}</p>}
      </Card>
      <div className="grid g2 mt">
        <Card title="F1 by system">
          <HBar rows={sys.map(([k, s]) => ({ label: k, value: s.f1, ci: s.ci95?.f1, hl: isFull(k), color: isFull(k) ? null : "var(--baseline)" }))} fmt={f2} labelW={210} max={1} valueLabel="F1" />
        </Card>
        <Card title="Comments a reviewer reads per real issue" sub="Lower is better.">
          <HBar rows={sys.filter(([, s]) => s.comments_per_real_issue).map(([k, s]) => ({ label: k, value: s.comments_per_real_issue, hl: isFull(k), color: isFull(k) ? null : "var(--baseline)" }))}
            fmt={f2} labelW={210} valueLabel="Comments / real issue" />
        </Card>
      </div>
      {runBox}
    </>
  );
}
