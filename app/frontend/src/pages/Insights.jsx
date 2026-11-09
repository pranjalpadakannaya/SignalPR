import { HBar, Lines, VBar } from "../charts/index.jsx";
import { Card, Chip, Loading, NotReady, PageHead, Tile } from "../components/ui.jsx";
import { AGENT_ORDER, agentVar, num, pct, prettyAgent, useApi } from "../lib.js";

const kfmt = (v) => (v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k" : String(Math.round(v)));

export default function Insights() {
  const { data: d, error, loading } = useApi("/api/insights");
  if (loading) return <Loading />;
  if (error) return <NotReady title="No mined dataset yet" cmd="python -m risk mine && python -m risk report" />;
  const t = d.totals;
  const months = d.monthly.map((m) => m.month);
  const agents = AGENT_ORDER.filter((a) => d.monthly.some((m) => m[a]));
  return (
    <>
      <PageHead title="What human reviewers comment on">
        Mined from <b>{d.source}</b>{d.synthetic && <> <Chip>synthetic sample</Chip></>}: every pull request opened by an AI coding agent
        in a popular repository, joined to the reviews and inline comments humans left on it ({t.date_min.slice(0, 10)} → {t.date_max.slice(0, 10)}).
      </PageHead>
      <div className="grid g5">
        <Tile label="Agent pull requests" value={num(t.prs)} sub={`${num(t.repos)} repositories · ${t.agents} agents`} />
        <Tile label="Changed files" value={num(t.file_changes)} sub="one row per (PR, file)" />
        <Tile label="Human inline comments" value={num(t.inline_comments_human)} sub={`${num(t.inline_comments_bot)} more from bots`} />
        <Tile label="Training population" value={`${num(t.population_files)}`} sub={`files in ${num(t.population_prs)} merged, human-reviewed PRs`} />
        <Tile label="Files that drew a comment" value={pct(t.positive_rate, 2)} sub={`${num(t.positive_files)} positives (the label)`} />
      </div>
      <div className="grid g2 mt">
        <Card title="Agent pull requests per month">
          <Lines series={agents.map((a) => ({ name: prettyAgent(a), color: agentVar(a), points: d.monthly.map((m, i) => [i, m[a] || 0]) }))}
            categories={months} yFmt={kfmt} height={290} />
        </Card>
        <Card title="How each agent's PRs fare">
          <table className="data"><thead><tr><th>Agent</th><th className="num">PRs</th><th className="num">Merged</th><th className="num">Human-reviewed</th><th className="num">Got inline comments</th></tr></thead>
            <tbody>{d.agents.map((a) => <tr key={a.agent}>
              <td><span className="legend" style={{ display: "inline-flex", margin: 0 }}><span><i className="sq" style={{ background: agentVar(a.agent) }} />{prettyAgent(a.agent)}</span></span></td>
              <td className="num">{num(a.prs)}</td><td className="num">{pct(a.merge_rate)}</td><td className="num">{pct(a.human_review_rate)}</td>
              <td className="num">{pct(a.inline_comment_pr_rate)}</td></tr>)}</tbody></table>
          <h3 className="mt">Merge rate by agent</h3>
          <HBar rows={d.agents.map((a) => ({ label: prettyAgent(a.agent), value: a.merge_rate, color: agentVar(a.agent), sub: `${num(a.prs)} PRs` }))}
            fmt={(v) => pct(v, 0)} max={1} valueLabel="Merged" />
        </Card>
      </div>
      <div className="grid g2 mt">
        <Card title="Bigger changes draw more comments" sub="Share of files commented on, by lines changed in the file (PRs that received inline review).">
          <VBar rows={d.rate_by_churn.map((r) => ({ label: r.key, value: r.rate, n: r.n }))} fmt={(v) => pct(v, 0)} xLabel="Lines changed in file" valueLabel="Commented" />
        </Card>
        <Card title="…but not every file type equally" sub="Share of files commented on, by file category.">
          <HBar rows={d.rate_by_category.map((r) => ({ label: r.key, value: r.rate, sub: `n = ${num(r.n)}` }))} fmt={(v) => pct(v, 1)} valueLabel="Commented" />
        </Card>
      </div>
      <div className="grid g2 mt">
        <Card title="By language"><HBar rows={d.rate_by_lang.map((r) => ({ label: r.key, value: r.rate, sub: `n = ${num(r.n)}` }))} fmt={(v) => pct(v, 1)} valueLabel="Commented" /></Card>
        <Card title="What reviewers talk about" sub="Share of human inline comments mentioning each topic (keyword match; a comment can hit several).">
          <HBar rows={d.comment_topics.map((r) => ({ label: r.topic, value: r.share }))} fmt={(v) => pct(v, 0)} valueLabel="Share of comments" />
        </Card>
      </div>
      <div className="grid g2 mt">
        <Card title="By number of files in the PR" sub="Share of files commented on across all merged, human-reviewed PRs: in big PRs each file gets less attention.">
          <VBar rows={d.rate_by_pr_size.map((r) => ({ label: r.key, value: r.rate, n: r.n }))} fmt={(v) => pct(v, 0)} xLabel="Files in PR" valueLabel="Commented" />
        </Card>
        <Card title="By agent (file level)">
          <HBar rows={d.rate_by_agent.map((r) => ({ label: prettyAgent(r.key), value: r.rate, color: agentVar(r.key), sub: `n = ${num(r.n)}` }))} fmt={(v) => pct(v, 1)} valueLabel="Commented" />
        </Card>
      </div>
    </>
  );
}
