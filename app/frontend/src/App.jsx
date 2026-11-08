import { useCallback, useEffect, useState } from "react";
import { TooltipProvider } from "./charts/index.jsx";
import { Dot } from "./components/ui.jsx";
import { api, f3, num, useHashRoute } from "./lib.js";
import { StatusCtx } from "./status.js";
import Evaluation from "./pages/Evaluation.jsx";
import History from "./pages/History.jsx";
import Insights from "./pages/Insights.jsx";
import Model from "./pages/Model.jsx";
import Review from "./pages/Review.jsx";
import Setup from "./pages/Setup.jsx";


const I = {
  review: <path d="M4 5h16v11H8l-4 4z" />,
  insights: <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />,
  model: <><circle cx="12" cy="12" r="3" /><path d="M12 2v4M12 18v4M2 12h4M18 12h4M5 5l3 3M16 16l3 3M5 19l3-3M16 8l3-3" /></>,
  eval: <><path d="M9 11l3 3 8-8" /><path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9" /></>,
  history: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  setup: <><circle cx="12" cy="12" r="3" /><path d="M12 1v3M12 20v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M1 12h3M20 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" /></>,
};
const NAV = [["review", "Review a PR"], ["insights", "Data insights"], ["model", "Risk model"], ["eval", "Evaluation"],
  ["history", "History"], ["setup", "Setup & AI reviewers"]];
const PAGES = { review: Review, insights: Insights, model: Model, eval: Evaluation, history: History, setup: Setup };

function Sidebar({ page, status }) {
  const toggleTheme = () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark";
    try { localStorage.setItem("signalpr-theme", document.documentElement.dataset.theme); } catch {}
  };
  const s = status;
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round"><path d="M3 16l5-6 4 4 8-10" /></svg></div>
        <div><div className="brand-name">SignalPR</div><div className="brand-sub">Review what matters</div></div>
      </div>
      <nav className="nav">
        {NAV.map(([k, label]) => (
          <a key={k} href={`#/${k}`} className={page === k ? "active" : ""}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">{I[k]}</svg>{label}</a>
        ))}
      </nav>
      <div className="sidebar-foot">
        {!s ? <div className="status-line"><Dot ok={false} />Connecting…</div> : <>
          <a className="status-line" href="#/setup" style={{ color: "inherit" }}>
            <Dot ok={s.llm.available ? true : "warn"} />{s.llm.available ? `AI: ${s.llm.model}` : "AI reviewers off · set up"}</a>
          <div className="status-line"><Dot ok={s.risk_model ? true : "warn"} />
            {s.risk_model ? `Risk model · PR-AUC ${f3(s.risk_model.test_pr_auc)}` : "Risk model not trained"}</div>
          <div className="status-line"><Dot ok={s.data.source ? (s.data.synthetic ? "warn" : true) : "warn"} />
            {s.data.source ? `AIDev · ${num(s.data.totals?.prs)} PRs` : "No mined data"}</div>
        </>}
        <button className="theme-toggle" onClick={toggleTheme}>◐ Theme</button>
      </div>
    </aside>
  );
}

export default function App() {
  const [page = "review", arg] = useHashRoute();
  const [status, setStatus] = useState(null);
  const refresh = useCallback(() => api("/api/status").then(setStatus).catch(() => setStatus(null)), []);
  useEffect(() => { refresh(); }, [refresh]);
  const Page = PAGES[page] || Review;
  return (
    <StatusCtx.Provider value={{ status, refresh }}>
      <TooltipProvider>
        <div className="app">
          <Sidebar page={page} status={status} />
          <main><Page arg={arg} key={page} /></main>
        </div>
      </TooltipProvider>
    </StatusCtx.Provider>
  );
}
