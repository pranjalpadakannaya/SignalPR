"""LLM provider settings edited from the dashboard's Setup page.

Saved to .cache/llm_settings.json (git-ignored, readable only by you). SIGNALPR_LLM_* environment
variables win when they are set, so scripted runs on Sol or Kaggle are never changed by the UI.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from agent import llm
from agent.llm_client import ROOT, LLMConfig

SETTINGS_PATH = Path(os.environ.get("SIGNALPR_SETTINGS", ROOT / ".cache" / "llm_settings.json"))

PRESETS = {
    "ollama": {
        "label": "Ollama (local, free)",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5-coder:7b",
        "needs_key": False,
        "key_url": "https://ollama.com/download",
        "models": ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen2.5-coder:32b"],
        "note": "Runs on your machine or a Kaggle GPU (scripts/serve/). Pull the model first.",
    },
    "sol": {
        "label": "vLLM on ASU Sol",
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "needs_key": False,
        "key_url": "",
        "models": ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct"],
        "note": "Start scripts/serve/sol_vllm.sbatch, then tunnel the port (ssh -L 8000:<node>:8000 sol).",
    },
    "groq": {
        "label": "Groq (free tier)",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3-32b",
        "needs_key": True,
        "key_url": "https://console.groq.com/keys",
        "models": ["qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"],
        "note": "Hosted open-weight models with a free, rate-limited key. Handy for demos without a GPU.",
    },
    "custom": {
        "label": "Other OpenAI-compatible server",
        "base_url": "http://localhost:8001/v1",
        "model": "",
        "needs_key": False,
        "key_url": "",
        "models": [],
        "note": "LM Studio, OpenRouter, or any /v1 server.",
    },
    "off": {
        "label": "Off (tool evidence only)",
        "base_url": "",
        "model": "",
        "needs_key": False,
        "key_url": "",
        "models": [],
        "note": "Specialists turn analyzer findings into comments; no model calls.",
    },
}

ENV_OVERRIDE = any(os.environ.get(k) for k in ("SIGNALPR_LLM_BASE_URL", "SIGNALPR_LLM_MODEL", "SIGNALPR_LLM_PRESET"))


def load() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def apply(s: dict | None = None) -> None:
    """Point the shared LLM client at the saved provider."""
    s = load() if s is None else s
    if ENV_OVERRIDE or not s:
        return
    if s.get("provider") == "off":
        os.environ["SIGNALPR_LLM"] = "off"
        llm.configure()
        return
    os.environ.pop("SIGNALPR_LLM", None)
    cfg = replace(LLMConfig.from_env(), base_url=s["base_url"], model=s["model"], api_key=s.get("api_key") or "none")
    llm.configure(cfg)


def save(provider: str, base_url: str, model: str, api_key: str | None) -> dict:
    old = load()
    if not api_key:  # keep the saved key when the field is left blank
        api_key = old.get("api_key", "") if old.get("provider") == provider else ""
    s = {"provider": provider, "base_url": (base_url or "").rstrip("/"), "model": model or "", "api_key": api_key}
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=1))
    try:
        os.chmod(SETTINGS_PATH, 0o600)
    except OSError:
        pass
    apply(s)
    return s


def public() -> dict:
    s, c = load(), llm.client().config
    key = s.get("api_key", "")
    return {
        "provider": s.get("provider") or ("env" if ENV_OVERRIDE else None),
        "base_url": c.base_url,
        "model": c.model,
        "has_key": bool(key),
        "key_hint": (key[:4] + "…" + key[-4:]) if len(key) > 10 else "",
        "env_override": ENV_OVERRIDE,
    }
