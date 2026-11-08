import { useEffect, useRef, useState } from "react";
import { Callout, Card, Cmd, Dot, Loading, PageHead } from "../components/ui.jsx";
import { api, f3, post } from "../lib.js";
import { useStatus } from "../status.js";

const ORDER = ["ollama", "sol", "groq", "custom", "off"];

export default function Setup() {
  const { status, refresh } = useStatus();
  const [llm, setLlm] = useState(null);
  const [form, setForm] = useState(null);
  const [busy, setBusy] = useState("");
  const [test, setTest] = useState(null);
  const timer = useRef();

  const load = async () => {
    const d = await api("/api/llm");
    setLlm(d);
    if (!form) {
      const prov = d.settings.provider && d.presets[d.settings.provider] ? d.settings.provider : "ollama";
      const p = d.presets[prov];
      setForm({ provider: prov, base_url: d.settings.provider === prov ? d.settings.base_url : p.base_url,
        model: d.settings.provider === prov ? d.settings.model : p.model, api_key: "" });
    }
    if (Object.values(d.pulls || {}).some((x) => x.state === "running")) timer.current = setTimeout(load, 3000);
  };
  useEffect(() => { load(); return () => clearTimeout(timer.current); }, []);
  if (!llm || !form) return <Loading />;

  const P = llm.presets, p = P[form.provider], st = llm.status, s = llm.settings;
  const pick = (prov) => { setTest(null); setForm({ provider: prov, base_url: P[prov].base_url, model: P[prov].model, api_key: "" }); };
  const save = async () => {
    setBusy("Saving and testing…"); setTest(null);
    try { const r = await post("/api/llm", form); setTest(r.test); await load(); refresh(); }
    catch (e) { setTest({ ok: false, error: e.message }); }
    setBusy("");
  };
  const pull = async () => { await post("/api/llm/pull", { model: form.model }); load(); };
  const pullState = llm.pulls?.[form.model];
  const saved = s.provider === form.provider;
  const ollamaMissingModel = form.provider === "ollama" && st.available && st.models?.length >= 0 && !st.models?.includes(form.model);

  return (
    <>
      <PageHead title="Setup & AI reviewers">Everything SignalPR uses is free: open-weight models, open-source analyzers, a public dataset.</PageHead>
      <div className="split">
        <div>
          <Card title="AI reviewers (LLM)" sub="The intent checker and the three specialist reviewers use any OpenAI-compatible model. Pick a provider, paste a key if it needs one, and save. SignalPR tests the connection right away.">
            {s.env_override && <Callout kind="warn">Environment variables (SIGNALPR_LLM_BASE_URL / _MODEL / _PRESET) are set, so they override what you save here.</Callout>}
            <div className="sample-list mt-s">
              {ORDER.map((k) => (
                <button key={k} className={`sample ${form.provider === k ? "selected" : ""}`} onClick={() => pick(k)}>
                  <div className="t">{P[k].label}</div><div className="small muted mt-s">{P[k].note}</div>
                  {s.provider === k && <div className="small mt-s" style={{ color: "var(--good-ink)" }}>● current</div>}
                </button>))}
            </div>
            {form.provider !== "off" && <div className="mt">
              {p.needs_key && <div className="field">
                <label>API key {p.key_url && <a href={p.key_url} target="_blank" rel="noopener">(get a free key ↗)</a>}</label>
                <input type="password" value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })}
                  placeholder={saved && s.has_key ? `saved key ${s.key_hint}. Leave blank to keep it.` : "paste your key"} />
                <div className="small muted mt-s">Saved only on this computer, in .cache/llm_settings.json.</div>
              </div>}
              <div className="row">
                <div className="grow"><label>Model</label>
                  <input type="text" list="models" value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} />
                  <datalist id="models">{(form.provider === "ollama" && st.models?.length ? st.models : p.models).map((m) => <option key={m} value={m} />)}</datalist></div>
                <div className="grow"><label>Server URL</label>
                  <input type="text" value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} /></div>
              </div>
            </div>}
            <div className="row mt">
              <button className="btn primary" onClick={save} disabled={!!busy || (p.needs_key && !form.api_key && !(saved && s.has_key))}>
                {busy || (form.provider === "off" ? "Turn AI reviewers off" : "Save & test connection")}</button>
              {form.provider === "ollama" && <button className="btn" onClick={pull} disabled={pullState?.state === "running"}>
                {pullState?.state === "running" ? `Downloading ${form.model}…` : `Download ${form.model} in Ollama`}</button>}
            </div>
            {test && (test.ok ? <Callout className="mt"><b style={{ color: "var(--good-ink)" }}>✓ Connected.</b> The model answered in {test.seconds}s. AI reviewers are on; go to <a href="#/review">Review a PR</a>.</Callout>
              : <Callout kind="err" className="mt"><b>Connection failed.</b> {test.error}</Callout>)}
            {pullState?.state === "error" && <Callout kind="err" className="mt">Download failed: {pullState.error}</Callout>}
            {pullState?.state === "done" && <Callout className="mt">Model downloaded. Click <b>Save &amp; test connection</b>.</Callout>}
            {form.provider === "ollama" && !st.available && <Callout kind="warn" className="mt">Ollama isn't running. Install it from ollama.com (or <code>brew install ollama</code>), then run <code>ollama serve</code> in a terminal.</Callout>}
            {ollamaMissingModel && st.models && <div className="small muted mt-s">Models installed in Ollama: {st.models.join(", ") || "none yet"}.</div>}
          </Card>
          <Card title="GitHub App (automatic reviews on real PRs)">
            <ol className="small secondary" style={{ paddingLeft: 18, margin: 0 }}>
              <li>GitHub → Settings → Developer settings → GitHub Apps → <b>New GitHub App</b>.</li>
              <li>Webhook URL <code>https://&lt;your-host&gt;/webhook/github</code> (local testing: <code>ngrok http 8000</code>). Set a webhook secret.</li>
              <li>Permissions: <b>Pull requests: Read &amp; write</b>, <b>Contents: Read</b>. Subscribe to <b>Pull request</b> events.</li>
              <li>Generate a private key, install the app on a repo, then:</li></ol>
            <Cmd>{`pip install cryptography   # needed for GitHub App sign-in
export GITHUB_APP_ID=123456
export GITHUB_APP_PRIVATE_KEY_PATH=~/signalpr.private-key.pem
export GITHUB_WEBHOOK_SECRET=…
python -m app`}</Cmd>
            <p className="small secondary">Simpler: <code>export GITHUB_TOKEN=ghp_…</code> and use <b>Post to GitHub</b> on any review.</p>
          </Card>
        </div>
        <div>
          <Card title="Status">
            <table className="data"><tbody>
              <tr><td><Dot ok={st.available ? true : "warn"} /> AI reviewers</td><td className="small">{st.available ? `${st.model}` : (s.provider ? st.reason || "off" : "not set up yet (pick a provider)")}</td></tr>
              <tr><td><Dot ok={status?.risk_model ? true : "warn"} /> Risk model</td><td className="small">{status?.risk_model ? `${status.risk_model.backend} · PR-AUC ${f3(status.risk_model.test_pr_auc)}` : "not trained"}</td></tr>
              <tr><td><Dot ok={!!status?.data?.source} /> Dataset</td><td className="small">{status?.data?.source || "not mined"}</td></tr>
              <tr><td><Dot ok={status?.github?.token ? true : "warn"} /> GitHub token</td><td className="small">{status?.github?.token ? "configured" : "not set (public PRs, 60 requests/hour)"}</td></tr>
              <tr><td><Dot ok={status?.github?.app ? true : "warn"} /> GitHub App</td><td className="small">{status?.github?.app ? "configured" : "not configured"}</td></tr>
              <tr><td><Dot ok /> Server</td><td className="small">{status?.framework} · v{status?.version}</td></tr>
            </tbody></table>
          </Card>
          <Card title="Rebuild data and model">
            <Cmd>{`python -m risk all          # download → mine → train → report (results/risk/)
python -m eval --bench demo # benchmark + ablations (results/eval/)`}</Cmd>
          </Card>
          <Card title="Change the UI (React)">
            <Cmd>{`cd app/frontend
npm install
npm run dev      # rebuilds on save`}</Cmd>
          </Card>
        </div>
      </div>
    </>
  );
}
