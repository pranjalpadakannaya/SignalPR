"""SQLite history of reviews run from the dashboard, the CLI or the GitHub App webhook."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from agent.llm_client import ROOT

DB_PATH = Path(os.environ.get("SIGNALPR_DB", ROOT / ".cache" / "reviews.sqlite3"))
_lock = threading.Lock()
COLS = ["id", "created", "title", "repo", "url", "source", "mode", "shown", "dropped", "critical"]


def _conn(path: Path | None = None):
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute(
        "CREATE TABLE IF NOT EXISTS reviews (id TEXT PRIMARY KEY, created REAL, title TEXT, repo TEXT, url TEXT, "
        "source TEXT, mode TEXT, shown INT, dropped INT, critical INT, result TEXT)"
    )
    return c


def save(result: dict, source: str) -> str:
    rid = uuid.uuid4().hex[:10]
    p, st = result["pr"], result["stats"]
    row = (
        rid,
        time.time(),
        p["title"],
        p["repo"],
        p["url"],
        source,
        result["options"]["mode"],
        st["shown"],
        st["dropped"],
        st["severity"]["critical"],
        json.dumps(result, default=str),
    )
    with _lock, _conn() as c:
        c.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
    return rid


def list_reviews(limit: int = 100) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(f"SELECT {', '.join(COLS)} FROM reviews ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(COLS, r)) for r in rows]


def get(rid: str) -> dict | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT result FROM reviews WHERE id=?", (rid,)).fetchone()
    return json.loads(row[0]) if row else None


def delete(rid: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM reviews WHERE id=?", (rid,))
