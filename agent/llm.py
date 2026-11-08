"""The LLM handle the review steps share: one cached LLMClient (issue #2) per process.

Steps call `chat_json(system, user, ...)`; `available()` tells the pipeline whether to run the
LLM steps or fall back to tool-only comments. Set SIGNALPR_LLM=off to force the offline path.
"""

from __future__ import annotations

import os
import threading
import time

from agent.llm_client import LLMClient, LLMConfig, LLMError, extract_json

_lock = threading.Lock()
_client: LLMClient | None = None
_status: dict = {"t": 0.0}


def configure(config: LLMConfig | None = None) -> LLMClient:
    """Replace the shared client (the web app's settings page and the eval harness use this)."""
    global _client
    with _lock:
        _client = LLMClient(config or LLMConfig.from_env())
        _status["t"] = 0.0
    return _client


def client() -> LLMClient:
    return _client or configure()


def status(force: bool = False) -> dict:
    """Is the LLM server reachable? Cached for 30 s."""
    if os.environ.get("SIGNALPR_LLM", "auto").lower() == "off":
        return {"available": False, "reason": "disabled (SIGNALPR_LLM=off)", "model": client().config.model}
    c = client()
    if c.config.cache_mode == "only":  # replaying cached outputs needs no server
        return {"available": True, "reason": "replay from cache", "model": c.config.model}
    if not force and time.time() - _status["t"] < 30:
        return {k: v for k, v in _status.items() if k != "t"}
    try:
        h = c.health()
        res = {"available": True, "reason": "", "model": c.config.model, "models": h["models"]}
        if h["models"] and c.config.model not in h["models"]:
            res["reason"] = f"model {c.config.model} is not in the server's list"
    except LLMError as e:
        res = {"available": False, "reason": str(e)[:200], "model": c.config.model}
    _status.clear()
    _status.update(res, t=time.time())
    return res


def available() -> bool:
    return status()["available"]


def chat_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    sample: int = 0,
    tag: str | None = None,
    max_tokens: int | None = None,
):
    """Send one system+user prompt, return (parsed JSON or None, raw text)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    r = client().chat(msgs, temperature=temperature, sample=sample, json_mode=True, tag=tag, max_tokens=max_tokens)
    return extract_json(r.text), r.text
