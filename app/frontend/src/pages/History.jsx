import { Card, Chip, Empty, Loading, PageHead } from "../components/ui.jsx";
import { api, useApi } from "../lib.js";

export default function History() {
  const { data: rows, loading, reload } = useApi("/api/reviews");
  if (loading) return <Loading />;
  const del = async (e, id) => { e.stopPropagation(); await api(`/api/reviews/${id}`, { method: "DELETE" }); reload(); };
  return (
    <>
      <PageHead title="Review history">Every review run from the dashboard, the CLI or the GitHub App webhook.</PageHead>
      <Card>
        {!rows?.length ? <Empty>No reviews yet. <a href="#/review">Review a PR</a>.</Empty> : (
          <table className="data"><thead><tr><th>When</th><th>Pull request</th><th>Source</th><th>System</th>
            <th className="num">Shown</th><th className="num">Filtered</th><th className="num">Critical</th><th /></tr></thead>
            <tbody>{rows.map((r) => (
              <tr key={r.id} className="click" onClick={() => (location.hash = `#/review/${r.id}`)}>
                <td className="small">{new Date(r.created * 1000).toLocaleString()}</td>
                <td><b>{r.title}</b><div className="small muted">{r.repo}</div></td><td><Chip>{r.source}</Chip></td>
                <td className="small">{r.mode}</td><td className="num">{r.shown}</td><td className="num">{r.dropped}</td><td className="num">{r.critical}</td>
                <td><button className="btn sm" onClick={(e) => del(e, r.id)}>Delete</button></td>
              </tr>))}</tbody></table>
        )}
      </Card>
    </>
  );
}
