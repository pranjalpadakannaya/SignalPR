"""SignalPR web app: JSON API, React dashboard and GitHub App webhook.

    python -m app                      # http://localhost:8000
    python -m app --port 9000 --host 0.0.0.0

Runs on FastAPI (adds /docs) and falls back to plain Starlette, which FastAPI is built on.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from agent import github, llm
from agent.llm_client import ROOT
from agent.pipeline import Options, review
from agent.samples import list_samples, load_sample, pr_from_dict
from agent.state import PRInput
from app import settings, store
from risk.predict import load_model

try:
    from fastapi import FastAPI as App

    FRAMEWORK = "fastapi"
except ImportError:  # pragma: no cover
    from starlette.applications import Starlette as App

    FRAMEWORK = "starlette"

VERSION = "0.1.0"
DIST = Path(__file__).with_name("static")
RISK_RESULTS = ROOT / "results" / "risk"
EVAL_RESULTS = Path(os.environ.get("SIGNALPR_EVAL_RESULTS", ROOT / "results" / "eval" / "demo.json"))
STEPS = ["intent", "evidence", "specialists", "verifier", "ranker"]
JOBS: dict[str, dict] = {}
PULLS: dict[str, dict] = {}
EVAL = {"state": "idle"}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _not_ready(msg: str):
    return JSONResponse({"error": msg}, status_code=404)


# ------------------------------------------------------------------ status and results
async def status(request: Request):
    m = load_model()
    metrics = _read_json(RISK_RESULTS / "metrics.json") or {}
    stats = _read_json(RISK_RESULTS / "dataset_stats.json") or {}
    risk = None
    if m is not None:
        risk = {"backend": m.backend, "test_pr_auc": metrics.get("test", {}).get("pr_auc"), **(m.meta or {})}
    return JSONResponse(
        {
            "version": VERSION,
            "framework": FRAMEWORK,
            "llm": llm.status(force=True),
            "risk_model": risk,
            "data": {"source": stats.get("source"), "synthetic": stats.get("synthetic"), "totals": stats.get("totals")},
            "github": {
                "token": bool(os.environ.get("GITHUB_TOKEN")),
                "webhook_secret": bool(os.environ.get("GITHUB_WEBHOOK_SECRET")),
                "app": bool(os.environ.get("GITHUB_APP_ID") and os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH")),
            },
            "eval": EVAL_RESULTS.exists(),
        },
        headers={"Cache-Control": "no-store"},
    )


async def insights(request: Request):
    d = _read_json(RISK_RESULTS / "dataset_stats.json")
    return JSONResponse(d) if d else _not_ready("No mined dataset yet. Run `python -m risk mine`.")


async def model_metrics(request: Request):
    d = _read_json(RISK_RESULTS / "metrics.json")
    return JSONResponse(d) if d else _not_ready("No risk model results yet. Run `python -m risk train`.")


async def eval_results(request: Request):
    d = _read_json(EVAL_RESULTS)
    if not d:
        return _not_ready("No evaluation yet. Run `python -m eval --bench demo`.")
    for s in d.get("systems", {}).values():
        s.pop("per_pr", None)
    return JSONResponse(d)


def _run_eval(runs: int):
    from eval.run import run_benchmark

    EVAL.update(state="running", started=time.time(), progress="starting", error=None)
    try:
        run_benchmark("demo", repeats=1, llm_runs=runs, out=EVAL_RESULTS, progress=lambda m: EVAL.update(progress=m))
        EVAL.update(state="done")
    except Exception as e:  # shown on the Evaluation page
        EVAL.update(state="error", error=f"{type(e).__name__}: {e}")


async def eval_run(request: Request):
    b = await request.json() if await request.body() else {}
    if EVAL.get("state") != "running":
        threading.Thread(target=_run_eval, args=(int(b.get("runs", 3)),), daemon=True).start()
    return JSONResponse({"ok": True})


async def eval_status(request: Request):
    out = dict(EVAL)
    if out.get("started"):
        out["elapsed"] = round(time.time() - out["started"])
    return JSONResponse(out)


# ------------------------------------------------------------------ reviews
async def samples(request: Request):
    return JSONResponse(list_samples())


async def sample_detail(request: Request):
    try:
        return JSONResponse(load_sample(request.path_params["sid"]))
    except FileNotFoundError:
        return JSONResponse({"error": "not found"}, status_code=404)


def build_input(body: dict) -> tuple[PRInput, str]:
    kind = body.get("kind", "diff")
    if kind == "url":
        return github.fetch_pr(body["url"].strip(), token=body.get("token") or None), "github"
    if kind == "sample":
        return pr_from_dict(load_sample(body["sample_id"])), f"sample:{body['sample_id']}"
    pr = PRInput.from_diff(
        body.get("diff", ""),
        title=body.get("title") or "Pasted diff",
        description=body.get("description", ""),
        issue_title=body.get("issue_title", ""),
        issue_body=body.get("issue_body", ""),
        agent=body.get("agent") or "unknown",
        repo_language=body.get("repo_language") or None,
        stars=int(body.get("stars") or 0),
        repo_path=body.get("repo_path") or None,
        test_cmd=body.get("test_cmd") or None,
    )
    if not pr.files:
        raise ValueError("Could not parse a unified diff. Paste the output of `git diff` or a .patch file.")
    return pr, "diff"


def _run_job(jid: str, body: dict):
    job = JOBS[jid]
    try:
        job["phase"] = "fetching pull request" if body.get("kind") == "url" else "preparing"
        pr, src = build_input(body)
        job["phase"] = "reviewing"

        def progress(step, state):
            job["steps"][step] = state

        result = review(pr, Options.from_dict(body.get("options")), progress=progress)
        result["review_id"] = store.save(result, src)
        result["head_sha"], result["number"] = pr.head_sha, pr.number
        job.update(state="done", result=result, review_id=result["review_id"])
    except Exception as e:  # reported to the page that started the job
        job.update(state="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])


async def start_review(request: Request):
    body = await request.json()
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {
        "id": jid,
        "state": "running",
        "phase": "queued",
        "steps": {s: "pending" for s in STEPS},
        "log": [],
        "started": time.time(),
    }
    threading.Thread(target=_run_job, args=(jid, body), daemon=True).start()
    return JSONResponse({"job_id": jid})


async def job_status(request: Request):
    job = JOBS.get(request.path_params["jid"])
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    out = {k: v for k, v in job.items() if k != "result"}
    out["elapsed"] = round(time.time() - job["started"], 1)
    if job["state"] == "done":
        out["result"] = job["result"]
    return JSONResponse(out)


async def reviews(request: Request):
    return JSONResponse(store.list_reviews())


async def review_detail(request: Request):
    rid = request.path_params["rid"]
    if request.method == "DELETE":
        store.delete(rid)
        return JSONResponse({"ok": True})
    r = store.get(rid)
    return JSONResponse(r) if r else JSONResponse({"error": "not found"}, status_code=404)


async def post_to_github(request: Request):
    r = store.get(request.path_params["rid"])
    body = await request.json() if await request.body() else {}
    if not r or not r["pr"].get("url"):
        return JSONResponse({"error": "Only reviews of GitHub PR URLs can be posted."}, status_code=400)
    tok = body.get("token") or os.environ.get("GITHUB_TOKEN")
    if not tok:
        return JSONResponse({"error": "Set GITHUB_TOKEN (or paste a token) to post comments."}, status_code=400)
    try:
        pr = github.fetch_pr(r["pr"]["url"], token=tok, with_content=False)
        res = github.post_review(pr.repo, pr.number, pr.head_sha, r["comments"], r["summary_markdown"], token=tok)
        return JSONResponse({"ok": True, "html_url": res.get("html_url")})
    except github.GitHubError as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# ------------------------------------------------------------------ GitHub App webhook
REVIEW_ACTIONS = ("opened", "synchronize", "reopened", "ready_for_review")


def verify_signature(secret: str, raw: bytes, header: str) -> bool:
    good = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header or "", good)


def webhook_review(payload: dict):
    try:
        inst = (payload.get("installation") or {}).get("id")
        token = github.app_installation_token(inst) if inst and os.environ.get("GITHUB_APP_ID") else None
        token = token or os.environ.get("GITHUB_TOKEN")
        pr = github.fetch_pr(payload["pull_request"]["html_url"], token=token)
        result = review(pr)
        store.save(result, "webhook")
        if token:
            github.post_review(pr.repo, pr.number, pr.head_sha, result["comments"], result["summary_markdown"], token)
    except Exception:  # a failed background review must not take the server down
        traceback.print_exc()


async def webhook(request: Request):
    raw = await request.body()
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET")
    if secret and not verify_signature(secret, raw, request.headers.get("x-hub-signature-256", "")):
        return JSONResponse({"error": "bad signature"}, status_code=401)
    event = request.headers.get("x-github-event", "")
    payload = json.loads(raw or b"{}")
    if event == "ping":
        return JSONResponse({"ok": True, "pong": True})
    if event == "pull_request" and payload.get("action") in REVIEW_ACTIONS:
        if not payload.get("pull_request", {}).get("draft"):
            threading.Thread(target=webhook_review, args=(payload,), daemon=True).start()
            return JSONResponse({"ok": True, "queued": True}, status_code=202)
    return JSONResponse({"ok": True, "ignored": event})


# ------------------------------------------------------------------ LLM settings
async def llm_get(request: Request):
    return JSONResponse(
        {"settings": settings.public(), "presets": settings.PRESETS, "status": llm.status(force=True), "pulls": PULLS}
    )


def test_connection() -> dict:
    t = time.time()
    try:
        data, raw = llm.chat_json(
            "Reply with JSON only.", 'Return {"ok": true}', temperature=0, max_tokens=60, tag="setup/test"
        )
        return {"ok": True, "seconds": round(time.time() - t, 2), "reply": str(data if data is not None else raw)[:120]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}


async def llm_save(request: Request):
    b = await request.json()
    prov = b.get("provider", "custom")
    if prov not in settings.PRESETS:
        return JSONResponse({"error": f"unknown provider {prov}"}, status_code=400)
    p = settings.PRESETS[prov]
    settings.save(prov, b.get("base_url") or p["base_url"], b.get("model") or p["model"], b.get("api_key"))
    test = {"ok": True, "reply": "LLM off"} if prov == "off" else test_connection()
    return JSONResponse({"settings": settings.public(), "status": llm.status(force=True), "test": test})


async def llm_test(request: Request):
    return JSONResponse(test_connection())


def _ollama_pull(model: str):
    import urllib.request

    PULLS[model] = {"state": "running", "started": time.time()}
    base = llm.client().config.base_url.rstrip("/").removesuffix("/v1")
    req = urllib.request.Request(
        base + "/api/pull",
        data=json.dumps({"model": model, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:  # noqa: S310 - the user's own Ollama server
            r.read()
        PULLS[model] = {"state": "done"}
    except Exception as e:
        PULLS[model] = {"state": "error", "error": str(e)[:300]}


async def llm_pull(request: Request):
    model = (await request.json()).get("model") or llm.client().config.model
    if PULLS.get(model, {}).get("state") != "running":
        threading.Thread(target=_ollama_pull, args=(model,), daemon=True).start()
    return JSONResponse({"ok": True, "model": model})


# ------------------------------------------------------------------ pages
async def index(request: Request):
    if (DIST / "index.html").exists():
        return FileResponse(DIST / "index.html")
    return Response("Frontend not built. Run: cd app/frontend && npm install && npm run build", media_type="text/plain")


async def health(request: Request):
    return Response("ok")


async def figure(request: Request):
    p = RISK_RESULTS / "figures" / Path(request.path_params["name"]).name
    return FileResponse(p) if p.suffix == ".png" and p.is_file() else Response(status_code=404)


routes = [
    Route("/", index),
    Route("/health", health),
    Route("/api/status", status),
    Route("/api/insights", insights),
    Route("/api/model", model_metrics),
    Route("/api/eval", eval_results),
    Route("/api/eval/run", eval_run, methods=["POST"]),
    Route("/api/eval/status", eval_status),
    Route("/api/samples", samples),
    Route("/api/samples/{sid}", sample_detail),
    Route("/api/review", start_review, methods=["POST"]),
    Route("/api/jobs/{jid}", job_status),
    Route("/api/reviews", reviews),
    Route("/api/reviews/{rid}", review_detail, methods=["GET", "DELETE"]),
    Route("/api/reviews/{rid}/post", post_to_github, methods=["POST"]),
    Route("/api/llm", llm_get),
    Route("/api/llm", llm_save, methods=["POST"]),
    Route("/api/llm/test", llm_test, methods=["POST"]),
    Route("/api/llm/pull", llm_pull, methods=["POST"]),
    Route("/webhook/github", webhook, methods=["POST"]),
    Route("/figures/{name}", figure),
    Mount("/assets", app=StaticFiles(directory=str(DIST / "assets"), check_dir=False), name="assets"),
]


settings.apply()
app = App(routes=routes)
