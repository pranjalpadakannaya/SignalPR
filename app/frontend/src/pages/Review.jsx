import { useEffect, useRef, useState } from "react";
import { HBar } from "../charts/index.jsx";
import { AgentChip, Callout, Card, Chip, Code, PageHead, ScoreBar, Sev, Tile } from "../components/ui.jsx";
import { AGENT_ORDER, api, pct, post, prettyAgent, useApi } from "../lib.js";
import { useStatus } from "../status.js";

const STEPS = [["intent", "Intent checker"], ["evidence", "Evidence collector"], ["specialists", "Specialist reviewers"],
  ["verifier", "Verifier"], ["ranker", "Risk ranker"]];
const TOGGLES = [["intent", "Intent checker"], ["evidence", "Evidence collector"], ["verifier", "Verifier"],
  ["consistency", "Self-consistency"], ["ranker", "Risk ranker"]];
// keep form + last result when navigating away and back
const memory = { kind: "sample", sample: null, result: null, form: {} };

export default function Review({ arg }) {
  const { status, refresh } = useStatus();
  const samples = useApi("/api/samples");
  const [kind, setKind] = useState(memory.kind);
  const [sample, setSample] = useState(memory.sample);
  const [form, setForm] = useState({ url: "", token: "", title: "", agent: "unknown", lang: "", issue: "", diff: "",
    repo: "", test: "", ...memory.form });
  const [opts, setOpts] = useState({ mode: "signalpr", llm: "auto", runs: 3, max: 8, intent: true, evidence: true,
    verifier: true, consistency: true, ranker: true });
  const [job, setJob] = useState(null);
  const [result, setResult] = useState(memory.result);
  const [error, setError] = useState(null);
  const timer = useRef();
  const llmOk = status?.llm?.available;

  useEffect(() => { memory.kind = kind; memory.sample = sample; memory.form = form; }, [kind, sample, form]);
  useEffect(() => {
    if (!sample && samples.data?.length) setSample((samples.data.find((s) => s.known_issues) || samples.data[0]).id);
  }, [samples.data]);
  useEffect(() => {
    if (arg) api(`/api/reviews/${arg}`).then((r) => setResult({ ...r, review_id: arg })).catch((e) => setError(e.message));
  }, [arg]);
  useEffect(() => () => clearTimeout(timer.current), []);
  const f = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const poll = async (id) => {
    try {
      const j = await api(`/api/jobs/${id}`);
      setJob(j);
      if (j.state === "running") { timer.current = setTimeout(() => poll(id), 600); return; }
      if (j.state === "error") setError(j.error);
      else { setResult(j.result); memory.result = j.result; refresh(); }
    } catch (e) { setError(e.message); setJob(null); }
  };
  const start = async () => {
    setError(null); setResult(null);
    const body = { kind, options: { mode: opts.mode, use_llm: opts.llm === "off" ? false : null, runs: +opts.runs,
      max_comments: +opts.max, intent: opts.intent, evidence: opts.evidence, verifier: opts.verifier,
      consistency: opts.consistency, ranker: opts.ranker } };
    if (kind === "url") { if (!form.url.trim()) return setError("Paste a GitHub pull request URL."); Object.assign(body, { url: form.url.trim(), token: form.token.trim() }); }
    else if (kind === "diff") {
      if (!form.diff.trim()) return setError("Paste a unified diff.");
      Object.assign(body, { diff: form.diff, title: form.title, issue_body: form.issue, agent: form.agent,
        repo_language: form.lang, repo_path: form.repo, test_cmd: form.test });
    } else { if (!sample) return setError("Pick a demo PR."); body.sample_id = sample; }
    try { const { job_id } = await post("/api/review", body); setJob({ state: "running", steps: {}, phase: "queued", elapsed: 0, log: [] }); poll(job_id); }
    catch (e) { setError(e.message); }
  };
  const running = job?.state === "running";

  return (
    <>
      <PageHead title="Review a pull request">
        SignalPR checks the change against its issue, collects evidence from linters and analyzers, has three specialist
        reviewers draft comments, drops anything it cannot verify, and ranks the rest with a model trained on what human
        reviewers actually commented on.
      </PageHead>
      {status && !llmOk && (
        <Callout kind="warn" className="row" >
          <span style={{ flex: 1 }}><b>AI reviewers are off.</b> Reviews still work using analyzer evidence. Turn on the LLM specialists in
            about a minute with a free Groq key (no credit card), or a local Ollama model.</span>
          <a className="btn sm primary" href="#/setup">Turn on AI reviewers</a>
        </Callout>
      )}
      <Card className="mt">
        <div className="tabs">
          {[["url", "GitHub PR URL"], ["diff", "Paste a diff"], ["sample", "Demo PRs"]].map(([k, l]) =>
            <button key={k} className={`tab ${kind === k ? "active" : ""}`} onClick={() => setKind(k)}>{l}</button>)}
        </div>
        {kind === "url" && <div className="row">
          <div className="grow"><label>Pull request URL</label><input type="text" value={form.url} onChange={f("url")} placeholder="https://github.com/owner/repo/pull/123" /></div>
          <div style={{ width: 260 }}><label>GitHub token <span className="muted">(optional)</span></label>
            <input type="password" value={form.token} onChange={f("token")} placeholder={status?.github?.token ? "using GITHUB_TOKEN" : "ghp_… (private repos / rate limits)"} /></div>
        </div>}
        {kind === "diff" && <>
          <div className="row">
            <div className="grow"><label>PR title</label><input type="text" value={form.title} onChange={f("title")} placeholder="Add user search endpoint" /></div>
            <div style={{ width: 200 }}><label>Written by</label><select value={form.agent} onChange={f("agent")}>
              <option value="unknown">Human / unknown</option>{AGENT_ORDER.map((a) => <option key={a} value={a}>{prettyAgent(a)}</option>)}</select></div>
            <div style={{ width: 170 }}><label>Repo language</label><input type="text" value={form.lang} onChange={f("lang")} placeholder="Python" /></div>
          </div>
          <div className="field mt-s"><label>Issue / requirements <span className="muted">(used by the intent checker)</span></label>
            <textarea value={form.issue} onChange={f("issue")} style={{ minHeight: 70, fontFamily: "var(--sans)", fontSize: 13 }} placeholder="The admin page should let admins search users by name…" /></div>
          <div className="field"><label>Unified diff <span className="muted">(output of <code>git diff</code> or a .patch)</span></label>
            <textarea value={form.diff} onChange={f("diff")} placeholder={"diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,4 @@"} /></div>
          <details className="adv"><summary>Run the project's tests too (local checkout)</summary>
            <div className="row mt-s"><div className="grow"><label>Local repo path</label><input type="text" value={form.repo} onChange={f("repo")} placeholder="/Users/me/code/project" /></div>
              <div className="grow"><label>Test command</label><input type="text" value={form.test} onChange={f("test")} placeholder="pytest -q" /></div></div></details>
        </>}
        {kind === "sample" && <div className="sample-list">
          {(samples.data || []).map((s) => (
            <button key={s.id} className={`sample ${sample === s.id ? "selected" : ""}`} onClick={() => setSample(s.id)}>
              <div className="t">{s.title}</div><div className="small muted">{s.repo} · {prettyAgent(s.agent)}</div>
              <div className="small secondary mt-s">{s.known_issues ? `${s.known_issues} known issues planted` : "clean change (no planted issues)"}</div>
            </button>))}
        </div>}
        <details className="adv"><summary>Options</summary>
          <div className="row mt-s">
            <div style={{ width: 230 }}><label>System</label><select value={opts.mode} onChange={(e) => setOpts({ ...opts, mode: e.target.value })}>
              <option value="signalpr">SignalPR (full pipeline)</option>
              <option value="single_prompt" disabled={!llmOk}>Baseline: single LLM prompt</option>
              <option value="tools_only">Baseline: analysis tools only</option>
              <option value="size_rank">Baseline: rank by change size</option></select></div>
            <div style={{ width: 160 }}><label>AI reviewers</label><select value={opts.llm} onChange={(e) => setOpts({ ...opts, llm: e.target.value })}>
              <option value="auto">Auto {llmOk ? "(on)" : "(off)"}</option><option value="off">Off (tools only)</option></select></div>
            <div style={{ width: 130 }}><label>Runs (verifier)</label><input type="number" min="1" max="7" value={opts.runs} onChange={(e) => setOpts({ ...opts, runs: e.target.value })} /></div>
            <div style={{ width: 140 }}><label>Comments shown</label><input type="number" min="1" max="30" value={opts.max} onChange={(e) => setOpts({ ...opts, max: e.target.value })} /></div>
          </div>
          <div className="row mt-s">{TOGGLES.map(([k, l]) => (
            <label key={k} className="check"><input type="checkbox" checked={opts[k]} onChange={(e) => setOpts({ ...opts, [k]: e.target.checked })} /> {l}</label>))}</div>
          <div className="small muted mt-s">Unticking a step runs an ablation of the pipeline without it.</div>
        </details>
        <div className="row mt">
          <button className="btn primary" onClick={start} disabled={running}>{running ? "Reviewing…" : "Review pull request"}</button>
          <span className="small muted">{llmOk ? `AI reviewers: ${status.llm.model}` : "AI reviewers off: comments come from analyzer evidence."}</span>
        </div>
      </Card>
      {error && <Callout kind="err" className="mt"><b>Review failed.</b> {error}</Callout>}
      {job && (running || job.state === "error") && <Progress job={job} />}
      {result && <Result r={result} token={form.token} />}
    </>
  );
}

function Progress({ job }) {
  return (
    <div className="mt">
      <div className="stepper">{STEPS.map(([s, label], i) => {
        const st = job.steps?.[s] || "pending";
        return <div key={s} className={`step ${st}`}><div className="n">STEP {i + 1}</div>
          <div className="name">{label}</div><div className="st">{st === "pending" ? "waiting" : st}</div></div>;
      })}</div>
      <div className="small muted mt-s">{job.phase} · {job.elapsed}s{job.log?.length ? ` · ${job.log[job.log.length - 1]}` : ""}</div>
    </div>
  );
}

function CommentCard({ c, i }) {
  return (
    <div className="comment">
      <div className="comment-head"><span className="rank">{i + 1}</span><Sev s={c.severity} /><Chip>{c.category}</Chip>
        <span className="comment-title">{c.title}</span><span className="loc">{c.file}:{c.line ?? "–"}</span></div>
      <div className="comment-body"><p>{c.body}</p>
        {c.suggestion && <div className="suggestion"><b>Suggestion:</b> {c.suggestion}</div>}
        <Code snippet={c.snippet} line={c.line} /></div>
      <div className="comment-foot"><b style={{ color: "var(--text-primary)" }}>score {c.score.toFixed(2)}</b>
        <ScoreBar label="file risk" v={c.score_parts?.risk} /><ScoreBar label="severity" v={c.score_parts?.severity} />
        <ScoreBar label="evidence" v={c.score_parts?.evidence} />
        <span>{(c.verified_by || []).map((v) => <Chip key={v}>{v}</Chip>)}</span>
        <span className="muted" style={{ marginLeft: "auto" }}>{c.source}</span></div>
    </div>
  );
}

function Result({ r, token }) {
  const [copied, setCopied] = useState(false);
  const [posting, setPosting] = useState("");
  const pr = r.pr, st = r.stats;
  const copy = () => {
    const md = r.summary_markdown + "\n\n" + r.comments.map((c) => `#### \`${c.file}:${c.line}\`\n${c.markdown}`).join("\n\n");
    navigator.clipboard.writeText(md).then(() => setCopied(true));
  };
  const toGitHub = async () => {
    setPosting("Posting…");
    try { const res = await post(`/api/reviews/${r.review_id}/post`, { token }); setPosting("Posted ✓"); if (res.html_url) open(res.html_url, "_blank"); }
    catch (e) { setPosting(""); alert(e.message); }
  };
  return (
    <>
      <Card className="mt">
        <div className="pr-head"><div>
          <div className="pr-title">{pr.title || "Untitled change"}</div>
          <div className="small secondary mt-s">
            {pr.url && <><a href={pr.url} target="_blank" rel="noopener">{pr.repo}{pr.number ? ` #${pr.number}` : ""}</a> · </>}
            <AgentChip a={pr.agent} /> · {pr.files} files · <span className="diffstat"><span className="a">+{pr.additions}</span> <span className="d">−{pr.deletions}</span></span>
            {pr.issue_title && ` · issue: ${pr.issue_title}`}</div></div>
          <div className="row">
            {r.review_id && <a className="btn sm" href={`#/review/${r.review_id}`}>Permalink</a>}
            <button className="btn sm" onClick={copy}>{copied ? "Copied ✓" : "Copy as Markdown"}</button>
            {pr.url && <button className="btn sm primary" onClick={toGitHub} disabled={!!posting}>{posting || "Post to GitHub"}</button>}
          </div></div>
        <div className="grid g5 mt">
          <Tile label="Comments shown" value={st.shown} sub={st.more ? `+${st.more} lower priority` : "ranked by score"} />
          <Tile label="Filtered by verifier" value={st.dropped} sub="unverifiable or duplicate" />
          <Tile label="Critical" value={st.severity.critical} sub={["critical", "medium", "low"].map((s) => `${st.severity[s]} ${s}`).join(" · ")} />
          <Tile label="Evidence findings" value={st.evidence} sub="linters, analyzers, rules" />
          <Tile label="Review time" value={`${st.seconds}s`} sub={r.llm.used ? `AI: ${r.llm.model}` : "AI reviewers off"} />
        </div>
        {r.llm.errors?.length > 0 && <Callout kind="warn" className="mt">AI reviewer errors: {r.llm.errors.slice(0, 3).join(" | ")}</Callout>}
      </Card>
      <div className="split mt">
        <div>
          <Card title="Ranked review comments">
            {r.comments.length ? r.comments.map((c, i) => <CommentCard key={i} c={c} i={i} />)
              : <div className="empty">No verified issues. Nothing worth a reviewer's time was found.</div>}
            {r.more_comments.length > 0 && <details className="mt"><summary className="secondary" style={{ cursor: "pointer", fontWeight: 600 }}>
              Show {r.more_comments.length} lower-priority comment(s)</summary>
              <div className="mt-s">{r.more_comments.map((c, i) => <CommentCard key={i} c={c} i={i + r.comments.length} />)}</div></details>}
          </Card>
          <Card title={`Filtered out by the verifier (${r.dropped.length})`}
            sub="Candidates rejected because they are not on a changed line, have no supporting evidence, quote code that isn't there, appear in too few runs, or duplicate a better comment.">
            {r.dropped.length ? r.dropped.slice(0, 40).map((c, i) => (
              <div key={i} className="dropped"><b>{c.title}</b> <span className="loc">{c.file}:{c.line ?? "–"}</span> <span className="muted small">· {c.source}</span>
                <div className="reason">✕ {c.drop_reason}</div></div>)) : <div className="muted small">Nothing was filtered.</div>}
          </Card>
        </div>
        <div>
          <Card title="Intent check">
            <p className="small secondary">{r.intent.summary} <Chip>{r.intent.mode}</Chip>{r.intent.source && <span className="muted"> from {r.intent.source}</span>}</p>
            {(r.intent.requirements || []).length ? r.intent.requirements.map((q, i) => (
              <div key={i} className="req"><span className={`status ${q.status}`}>{q.status}</span><div className="txt">{q.text}
                {q.files?.length > 0 && <div className="small muted">{q.files.join(", ")}</div>}
                {q.evidence && <div className="small muted">{q.evidence}</div>}</div></div>))
              : <div className="muted small">No requirements found. Add an issue or description to check intent.</div>}
            {r.intent.unrequested?.length > 0 && <><h3 className="mt">Changes nobody asked for</h3>
              {r.intent.unrequested.map((u) => <div key={u.file} className="small"><span className="loc">{u.file}</span> <span className="muted">{u.reason}</span></div>)}</>}
          </Card>
          <Card title="File risk" sub={`Predicted chance a human reviewer comments on each file (${r.file_risk[0]?.source || ""}).`}>
            {r.file_risk.length > 0 && <HBar rows={r.file_risk.slice(0, 12).map((x) => ({ label: x.file.split("/").slice(-2).join("/"), value: x.risk, sub: `${x.file} · +${x.additions}/−${x.deletions}` }))}
              fmt={(v) => pct(v, 1)} labelW={150} valueW={54} valueLabel="P(reviewer comments)" />}
          </Card>
          <Card title="Pipeline trace">
            <table className="data"><tbody>{r.trace.map((t) => <tr key={t.step}><td>{t.step}</td><td className="small secondary">{t.detail}</td><td className="num">{t.ms} ms</td></tr>)}</tbody></table>
            <h3 className="mt">Tools</h3>
            <table className="data"><tbody>{Object.entries(r.tools || {}).map(([k, v]) => <tr key={k}><td>{k}</td><td className="small secondary">{v}</td></tr>)}</tbody></table>
          </Card>
        </div>
      </div>
    </>
  );
}
