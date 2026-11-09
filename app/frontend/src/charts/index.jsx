// Dependency-free SVG charts as React components: HBar (with CI whiskers), VBar, Grouped, Lines.
// Colours are CSS variables, so light/dark themes work automatically.
import { createContext, useContext, useState } from "react";
import { useWidth } from "../lib.js";

/* ---------------- tooltip ---------------- */
const TipCtx = createContext(() => {});
export function TooltipProvider({ children }) {
  const [tip, setTip] = useState(null);
  let style = { display: "none" };
  if (tip) {
    const x = tip.x + 280 > innerWidth ? tip.x - 270 : tip.x + 14;
    const y = tip.y + 120 > innerHeight ? tip.y - 100 : tip.y + 14;
    style = { display: "block", left: x, top: y };
  }
  return (
    <TipCtx.Provider value={setTip}>
      {children}
      <div className="tooltip" style={style}>{tip?.content}</div>
    </TipCtx.Provider>
  );
}
const useTip = () => {
  const set = useContext(TipCtx);
  return {
    show: (e, content) => set({ x: e.clientX, y: e.clientY, content }),
    hide: () => set(null),
  };
};

/* ---------------- helpers ---------------- */
export const niceMax = (v) => {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
};
const ticks = (max, n = 4) => Array.from({ length: n + 1 }, (_, i) => (max * i) / n);
// bar with a 4px rounded data end and a square baseline end
const barPath = (x, y, w, h, horizontal, r = 4) => {
  if (w <= 0 || h <= 0) return "";
  r = Math.max(0, Math.min(r, horizontal ? h / 2 : w / 2, horizontal ? w : h));
  return horizontal
    ? `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`
    : `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
};
const Swatch = ({ color, line, dash }) => (
  <i className={line ? "" : "sq"} style={{ background: dash ? `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 7px)` : color }} />
);
export const Legend = ({ items }) => (
  <div className="legend">{items.map((s) => <span key={s.name}><Swatch {...s} />{s.name}</span>)}</div>
);

/* ---------------- horizontal bars ---------------- */
export function HBar({ rows, fmt = (v) => v.toLocaleString(), max, labelW = 170, valueW = 70, rowH = 26,
  valueLabel = "Value", color = "var(--series-1)" }) {
  const [ref, width] = useWidth();
  const { show, hide } = useTip();
  const plotW = width - labelW - valueW;
  const m = max || niceMax(Math.max(...rows.map((r) => (r.ci ? r.ci[1] : r.value)), 1e-9));
  const height = rows.length * rowH + 8;
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        {rows.map((r, i) => {
          const y = i * rowH + 4, bh = Math.min(16, rowH - 8), w = Math.max(0, (r.value / m) * plotW);
          const ciEnd = r.ci ? (r.ci[1] / m) * plotW : 0;
          const label = r.label.length > 27 ? r.label.slice(0, 26) + "…" : r.label;
          return (
            <g key={r.label + i}>
              <text x={labelW - 10} y={y + bh / 2 + 4} textAnchor="end" style={r.hl ? { fill: "var(--text-primary)", fontWeight: 600 } : null}>{label}</text>
              <path d={barPath(labelW, y, w, bh, true)} fill={r.color || color} />
              {r.ci && (() => {
                const x1 = labelW + (r.ci[0] / m) * plotW, x2 = labelW + ciEnd, cy = y + bh / 2;
                return <g stroke="var(--text-primary)" strokeWidth="1.5">
                  <line x1={x1} x2={x2} y1={cy} y2={cy} /><line x1={x1} x2={x1} y1={cy - 4} y2={cy + 4} /><line x1={x2} x2={x2} y1={cy - 4} y2={cy + 4} />
                </g>;
              })()}
              <text className="val" x={labelW + Math.max(w, ciEnd) + 6} y={y + bh / 2 + 4}>{fmt(r.value)}</text>
              <rect className="hit" x="0" y={y - 4} width={width} height={rowH}
                onMouseMove={(e) => show(e, <><b>{r.label}</b><br />{valueLabel}: {fmt(r.value)}
                  {r.ci && <><br />95% CI: {fmt(r.ci[0])} – {fmt(r.ci[1])}</>}{r.sub && <><br /><span className="muted">{r.sub}</span></>}</>)}
                onMouseLeave={hide} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ---------------- shared vertical axis frame ---------------- */
function YGrid({ m, width, M, ph, fmt }) {
  return ticks(m).map((tv) => {
    const y = M.t + ph - (tv / m) * ph;
    return <g key={tv}><line className="gridline" x1={M.l} x2={width - M.r} y1={y} y2={y} />
      <text x={M.l - 8} y={y + 4} textAnchor="end">{fmt(tv)}</text></g>;
  });
}

/* ---------------- vertical bars ---------------- */
export function VBar({ rows, fmt = (v) => v.toLocaleString(), max, height = 230, xLabel, valueLabel = "Value", color = "var(--series-1)" }) {
  const [ref, width] = useWidth();
  const { show, hide } = useTip();
  const M = { t: 22, r: 10, b: 34, l: 46 }, pw = width - M.l - M.r, ph = height - M.t - M.b;
  const m = max || niceMax(Math.max(...rows.map((r) => r.value), 1e-9));
  const step = pw / rows.length, bw = Math.min(46, step * 0.62);
  return (
    <div className="chart" ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <YGrid m={m} width={width} M={M} ph={ph} fmt={fmt} />
        {rows.map((r, i) => {
          const x = M.l + i * step + (step - bw) / 2, h = (r.value / m) * ph, y = M.t + ph - h;
          return <g key={r.label}>
            <path d={barPath(x, y, bw, h, false)} fill={r.color || color} />
            <text x={x + bw / 2} y={height - M.b + 16} textAnchor="middle">{r.label}</text>
            <text className="val" x={x + bw / 2} y={y - 6} textAnchor="middle">{fmt(r.value)}</text>
            <rect className="hit" x={M.l + i * step} y={M.t} width={step} height={ph}
              onMouseMove={(e) => show(e, <><b>{xLabel ? `${xLabel}: ` : ""}{r.label}</b><br />{valueLabel}: {fmt(r.value)}{r.n != null && <><br />n = {r.n.toLocaleString()}</>}</>)}
              onMouseLeave={hide} />
          </g>;
        })}
        <line x1={M.l} x2={width - M.r} y1={M.t + ph} y2={M.t + ph} stroke="var(--border-strong)" />
        {xLabel && <text x={M.l + pw / 2} y={height - 2} textAnchor="middle">{xLabel}</text>}
      </svg>
    </div>
  );
}

/* ---------------- grouped bars ---------------- */
export function Grouped({ groups, series, fmt = (v) => v.toLocaleString(), max, height = 240 }) {
  const [ref, width] = useWidth();
  const { show, hide } = useTip();
  const M = { t: 22, r: 10, b: 30, l: 46 }, pw = width - M.l - M.r, ph = height - M.t - M.b;
  const m = max || niceMax(Math.max(...series.flatMap((s) => s.values), 1e-9));
  const step = pw / groups.length, inner = Math.min(step * 0.78, series.length * 34), bw = inner / series.length - 2;
  return (
    <div>
      <Legend items={series} />
      <div className="chart" ref={ref}>
        <svg viewBox={`0 0 ${width} ${height}`} role="img">
          <YGrid m={m} width={width} M={M} ph={ph} fmt={fmt} />
          {groups.map((g, gi) => {
            const x0 = M.l + gi * step + (step - inner) / 2;
            return <g key={g}>
              {series.map((s, si) => {
                const v = s.values[gi], h = (v / m) * ph, x = x0 + si * (bw + 2), y = M.t + ph - h;
                return <g key={s.name}>
                  <path d={barPath(x, y, bw, h, false)} fill={s.color} />
                  <rect className="hit" x={x} y={M.t} width={bw + 2} height={ph}
                    onMouseMove={(e) => show(e, <><b>{g}</b><br />{s.name}: {fmt(v)}</>)} onMouseLeave={hide} />
                </g>;
              })}
              <text x={M.l + gi * step + step / 2} y={height - M.b + 17} textAnchor="middle">{g}</text>
            </g>;
          })}
          <line x1={M.l} x2={width - M.r} y1={M.t + ph} y2={M.t + ph} stroke="var(--border-strong)" />
        </svg>
      </div>
    </div>
  );
}

/* ---------------- lines (numeric or categorical x) with crosshair ---------------- */
export function Lines({ series, height = 260, categories, xMin = 0, xMax, yMax, xLabel, yLabel,
  xFmt = (v) => v, yFmt = (v) => v, diagonal, hline }) {
  const [ref, width] = useWidth();
  const { show, hide } = useTip();
  const [hover, setHover] = useState(null);
  const M = { t: 14, r: 16, b: 40, l: 50 }, pw = width - M.l - M.r, ph = height - M.t - M.b;
  const xmin = categories ? 0 : xMin;
  const xmax = categories ? categories.length - 1 : (xMax ?? Math.max(...series.flatMap((s) => s.points.map((p) => p[0]))));
  const ymax = yMax ?? niceMax(Math.max(...series.flatMap((s) => s.points.map((p) => p[1]))));
  const X = (v) => M.l + ((v - xmin) / (xmax - xmin || 1)) * pw, Y = (v) => M.t + ph - (v / ymax) * ph;
  const every = categories ? Math.ceil(categories.length / 8) : 1;
  const xt = categories ? categories.map((c, i) => [i, c]).filter((_, i) => i % every === 0)
    : ticks(xmax - xmin).map((v) => [v + xmin, xFmt(v + xmin)]);
  const nearest = (s, xv) => s.points.reduce((b, p) => (Math.abs(p[0] - xv) < Math.abs(b[0] - xv) ? p : b), s.points[0]);
  const onMove = (e) => {
    const r = e.currentTarget.ownerSVGElement.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width) * width;
    const xv = xmin + ((sx - M.l) / pw) * (xmax - xmin);
    const pts = series.map((s) => nearest(s, xv));
    setHover({ sx, pts });
    const head = categories ? categories[Math.round(Math.max(0, Math.min(xmax, xv)))] : xLabel;
    show(e, <>{head && <><b>{head}</b><br /></>}{series.map((s, i) => (
      <div key={s.name}><Swatch color={s.color} /> {s.name}: <b>{yFmt(pts[i][1])}</b>{!categories && <span className="muted"> @ {xFmt(pts[i][0])}</span>}</div>))}</>);
  };
  return (
    <div>
      {series.length > 1 && <Legend items={series.map((s) => ({ ...s, line: true }))} />}
      <div className="chart" ref={ref}>
        <svg viewBox={`0 0 ${width} ${height}`} role="img">
          {ticks(ymax).map((tv) => <g key={tv}><line className="gridline" x1={M.l} x2={width - M.r} y1={Y(tv)} y2={Y(tv)} />
            <text x={M.l - 8} y={Y(tv) + 4} textAnchor="end">{yFmt(tv)}</text></g>)}
          {xt.map(([v, lab]) => <text key={v} x={X(v)} y={height - M.b + 16} textAnchor="middle">{lab}</text>)}
          <line x1={M.l} x2={width - M.r} y1={M.t + ph} y2={M.t + ph} stroke="var(--border-strong)" />
          {xLabel && <text x={M.l + pw / 2} y={height - 4} textAnchor="middle">{xLabel}</text>}
          {yLabel && <text x={12} y={M.t + ph / 2} textAnchor="middle" transform={`rotate(-90 12 ${M.t + ph / 2})`}>{yLabel}</text>}
          {diagonal && <line x1={X(0)} y1={Y(0)} x2={X(Math.min(xmax, ymax))} y2={Y(Math.min(xmax, ymax))} stroke="var(--baseline)" strokeDasharray="4 4" />}
          {hline != null && <line x1={M.l} x2={width - M.r} y1={Y(hline)} y2={Y(hline)} stroke="var(--baseline)" strokeDasharray="4 4" />}
          {series.map((s) => <path key={s.name} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeDasharray={s.dash ? "5 4" : "none"}
            d={s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")} />)}
          {series.filter((s) => s.markers).flatMap((s) => s.points.map((p, i) =>
            <circle key={s.name + i} cx={X(p[0])} cy={Y(p[1])} r="4" fill={s.color} stroke="var(--surface-1)" strokeWidth="2" />))}
          {hover && <>
            <line x1={hover.sx} x2={hover.sx} y1={M.t} y2={M.t + ph} stroke="var(--border-strong)" />
            {hover.pts.map((p, i) => <circle key={i} cx={X(p[0])} cy={Y(p[1])} r="4" fill={series[i].color} stroke="var(--surface-1)" strokeWidth="2" />)}
          </>}
          <rect className="hit" x={M.l} y={M.t} width={pw} height={ph} onMouseMove={onMove} onMouseLeave={() => { hide(); setHover(null); }} />
        </svg>
      </div>
    </div>
  );
}
