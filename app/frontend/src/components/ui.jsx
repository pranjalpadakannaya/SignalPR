import { prettyAgent } from "../lib.js";

export const Tile = ({ label, value, sub }) => (
  <div className="tile"><div className="label">{label}</div><div className="value">{value}</div>{sub && <div className="sub">{sub}</div>}</div>
);
export const Card = ({ title, sub, children, className = "", right }) => (
  <div className={`card ${className}`}>
    {(title || right) && <div className="row" style={{ justifyContent: "space-between", marginBottom: sub ? 4 : 12 }}>
      {title && <h2 style={{ margin: 0 }}>{title}</h2>}{right}</div>}
    {sub && <p className="small secondary">{sub}</p>}
    {children}
  </div>
);
const SEV = { critical: "Critical", medium: "Medium", low: "Low" };
export const Sev = ({ s }) => <span className={`sev ${s}`}><i />{SEV[s] || s}</span>;
export const Chip = ({ children, accent }) => <span className={`chip ${accent ? "accent" : ""}`}>{children}</span>;
export const AgentChip = ({ a }) => <Chip>{prettyAgent(a)}</Chip>;
export const Callout = ({ kind = "", children, className = "" }) => <div className={`callout ${kind} ${className}`}>{children}</div>;
export const Dot = ({ ok }) => <span className={`dot ${ok === true ? "ok" : ok === "warn" ? "warn" : "bad"}`} />;
export const Empty = ({ children }) => <div className="empty">{children}</div>;
export const PageHead = ({ title, children, right }) => (
  <div className="page-head"><div><h1>{title}</h1>{children && <p>{children}</p>}</div>{right}</div>
);
export const ScoreBar = ({ label, v = 0 }) => (
  <span className="scorebar" title={`${label}: ${Math.round(v * 100)}%`}>{label}
    <span className="track"><span className="fill" style={{ width: `${Math.round(v * 100)}%` }} /></span></span>
);
export function Code({ snippet, line }) {
  if (!snippet?.length) return null;
  return (
    <div className="code"><table><tbody>
      {snippet.map(([n, kind, t]) => (
        <tr key={n} className={`${kind === "+" ? "add" : ""} ${n === line ? "hl" : ""}`}>
          <td className="no">{n}</td><td className="tx">{kind === "+" ? "+" : " "} {t}</td></tr>))}
    </tbody></table></div>
  );
}
export const Cmd = ({ children }) => <pre className="cmd">{children}</pre>;
export const NotReady = ({ title, cmd }) => (
  <><PageHead title={title} /><Card><p>Run this from the project folder, then reload:</p><Cmd>{cmd}</Cmd></Card></>
);
export const Loading = () => <Empty>Loading…</Empty>;
