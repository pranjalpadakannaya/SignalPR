import hashlib
import hmac
import json
import time

import anyio
import pytest


async def _call(app, method, path, body=None, headers=None):
    """Minimal ASGI client (no httpx needed)."""
    raw = json.dumps(body).encode() if isinstance(body, (dict, list)) else (body or b"")
    hdrs = [(b"content-type", b"application/json")] + [(k.encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": hdrs,
        "server": ("test", 80),
        "client": ("test", 1),
        "root_path": "",
    }
    sent, out = False, {"status": None, "body": b""}

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            out["status"] = msg["status"]
        elif msg["type"] == "http.response.body":
            out["body"] += msg.get("body", b"")

    await app(scope, receive, send)
    try:
        return out["status"], json.loads(out["body"])
    except ValueError:
        return out["status"], out["body"]


@pytest.fixture()
def call(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNALPR_LLM", "off")
    monkeypatch.setattr("app.store.DB_PATH", tmp_path / "reviews.sqlite3")
    monkeypatch.setattr("app.settings.SETTINGS_PATH", tmp_path / "llm_settings.json")
    from app.server import app

    return lambda *a, **k: anyio.run(lambda: _call(app, *a, **k))


def test_status_and_pages(call):
    code, s = call("GET", "/api/status")
    assert code == 200 and s["llm"]["available"] is False
    assert call("GET", "/health") == (200, b"ok")
    code, samples = call("GET", "/api/samples")
    assert code == 200 and any(x["id"] == "flask-user-search" for x in samples)
    assert call("GET", "/api/samples/nope")[0] == 404


def test_review_job_and_history(call):
    code, j = call("POST", "/api/review", {"kind": "sample", "sample_id": "flask-user-search"})
    assert code == 200
    for _ in range(100):
        code, job = call("GET", f"/api/jobs/{j['job_id']}")
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done", job.get("error")
    assert set(job["steps"].values()) == {"done"}
    rid = job["result"]["review_id"]
    assert job["result"]["comments"]
    code, hist = call("GET", "/api/reviews")
    assert [h["id"] for h in hist] == [rid]
    assert call("GET", f"/api/reviews/{rid}")[1]["stats"]["shown"] == job["result"]["stats"]["shown"]
    assert call("POST", f"/api/reviews/{rid}/post", {})[0] == 400  # not a GitHub PR
    call("DELETE", f"/api/reviews/{rid}")
    assert call("GET", "/api/reviews")[1] == []


def test_bad_diff_reports_error(call):
    _, j = call("POST", "/api/review", {"kind": "diff", "diff": "not a diff"})
    for _ in range(50):
        _, job = call("GET", f"/api/jobs/{j['job_id']}")
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "error" and "unified diff" in job["error"]


def test_webhook_signature(call, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    body = json.dumps({"zen": "hi"}).encode()
    assert (
        call("POST", "/webhook/github", body, {"x-github-event": "ping", "x-hub-signature-256": "sha256=bad"})[0] == 401
    )
    sig = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
    code, r = call("POST", "/webhook/github", body, {"x-github-event": "ping", "x-hub-signature-256": sig})
    assert code == 200 and r["pong"]
    draft = json.dumps({"action": "opened", "pull_request": {"draft": True}}).encode()
    sig = "sha256=" + hmac.new(b"s3cret", draft, hashlib.sha256).hexdigest()
    code, r = call("POST", "/webhook/github", draft, {"x-github-event": "pull_request", "x-hub-signature-256": sig})
    assert code == 200 and r["ignored"] == "pull_request"


def test_llm_settings_roundtrip(call):
    code, d = call("GET", "/api/llm")
    assert code == 200 and "ollama" in d["presets"]
    code, d = call("POST", "/api/llm", {"provider": "off"})
    assert code == 200 and d["settings"]["provider"] == "off" and d["test"]["ok"]
    assert call("POST", "/api/llm", {"provider": "nope"})[0] == 400
