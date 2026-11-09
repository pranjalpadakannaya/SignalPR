// API client, formatting helpers, small hooks.
import { useCallback, useEffect, useRef, useState } from "react";

export async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || r.statusText), { status: r.status, data });
  return data;
}
export const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body ?? {}) });

export const pct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
export const num = (v) => (v == null ? "–" : Number(v).toLocaleString());
export const f3 = (v) => (v == null ? "–" : Number(v).toFixed(3));
export const f2 = (v) => (v == null ? "–" : Number(v).toFixed(2));

export const AGENT_ORDER = ["Claude_Code", "Copilot", "OpenAI_Codex", "Cursor", "Devin", "Google_Jules"];
export const agentVar = (a) => `var(--series-${((AGENT_ORDER.indexOf(a) + 8) % 8) + 1})`;
export const prettyAgent = (a) =>
  ({ Claude_Code: "Claude Code", OpenAI_Codex: "OpenAI Codex", Google_Jules: "Google Jules", unknown: "Human / unknown" }[a] || a);

/** Fetch JSON on mount (and when deps change). Returns {data, error, loading, reload}. */
export function useApi(path, deps = []) {
  const [state, set] = useState({ data: null, error: null, loading: true });
  const load = useCallback(() => {
    set((s) => ({ ...s, loading: true }));
    api(path).then((data) => set({ data, error: null, loading: false }))
      .catch((error) => set({ data: null, error, loading: false }));
  }, [path]);
  useEffect(load, [load, ...deps]);
  return { ...state, reload: load };
}

/** Track an element's width (for responsive SVG charts). */
export function useWidth(fallback = 600) {
  const ref = useRef(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Hash router: "#/review/abc" -> ["review", "abc"] */
export function useHashRoute() {
  const parse = () => (location.hash.replace(/^#\/?/, "") || "review").split("/");
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const on = () => { setRoute(parse()); window.scrollTo(0, 0); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return route;
}
