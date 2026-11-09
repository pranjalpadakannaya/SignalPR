# app/ — SignalPR web app and GitHub App

```bash
python -m app            # http://localhost:8000  (FastAPI; /docs lists the API)
```

Pages (React, source in `app/frontend/`, built bundle committed in `app/static/`):

| Page | Shows |
|---|---|
| Review a PR | Review a GitHub PR URL, a pasted diff, or a demo PR; live progress through the five steps; ranked comments with code, evidence and suggestion; filtered comments and why; intent checklist; file risk; post the review to GitHub |
| Data insights | AIDev-pop statistics from `results/risk/dataset_stats.json` |
| Risk model | Test metrics, baselines, PR curve, calibration, within-PR ranking, feature importance (`results/risk/metrics.json`) |
| Evaluation | SignalPR vs baselines and ablations (`results/eval/demo.json`); run the demo benchmark from the page |
| History | Every review run from the page, the CLI or the webhook (SQLite in `.cache/`) |
| Setup & AI reviewers | Pick the model server (Ollama, vLLM on Sol, Groq free tier, any OpenAI-compatible URL, or off), test it, pull an Ollama model |

## GitHub App (automatic reviews)

1. GitHub → Settings → Developer settings → GitHub Apps → New GitHub App.
2. Webhook URL `https://<host>/webhook/github` (local: `ngrok http 8000`), and a webhook secret.
3. Permissions: Pull requests read & write, Contents read. Subscribe to Pull request events.
4. Generate a private key, install the app on a repository, then

```bash
export GITHUB_APP_ID=123456
export GITHUB_APP_PRIVATE_KEY_PATH=~/signalpr.private-key.pem
export GITHUB_WEBHOOK_SECRET=...
python -m app --host 0.0.0.0
```

Opening or updating a non-draft PR then triggers a review that is posted as inline comments.
Without an App, `export GITHUB_TOKEN=...` and use **Post to GitHub** on any review of a PR URL.

## Change the UI

```bash
cd app/frontend
npm install
npm run dev      # rebuilds app/static on save; keep `python -m app` running
npm run build    # before committing
```
