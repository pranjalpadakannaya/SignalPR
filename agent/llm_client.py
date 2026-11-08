"""Shared LLM client for the agent and the baselines (issue #2).

Talks to any OpenAI-compatible server, so the same code works with
  * vLLM on ASU Sol            (scripts/serve/sol_vllm.sbatch)
  * Ollama on a laptop/Kaggle  (scripts/serve/ollama_local.sh, notebooks/serve_qwen_kaggle.ipynb)

Every prompt/response pair is cached on disk, keyed by model + prompt hash + settings (+ sample
index), so re-running an experiment costs nothing and gives identical outputs. Only the Python
standard library is used (urllib, sqlite3), so there is nothing extra to install.

    from agent.llm_client import LLMClient, LLMConfig

    llm = LLMClient(LLMConfig.from_preset("qwen-7b", backend="ollama"))
    r = llm.chat([{"role": "system", "content": "You review code."},
                  {"role": "user", "content": "Review this diff: ..."}])
    r.text, r.cached, r.latency_s, r.prompt_tokens, r.completion_tokens

    data = llm.chat_json(messages)            # parsed JSON (tolerates ```json fences / <think> blocks)
    for k in range(3):                        # self-consistency: 3 different samples, each cached separately
        llm.chat(messages, temperature=0.7, sample=k)

Config comes from agent/llm_models.yaml presets, a YAML file, or environment variables:
    SIGNALPR_LLM_PRESET=qwen-32b  SIGNALPR_LLM_BACKEND=vllm  SIGNALPR_LLM_BASE_URL=http://node:8000/v1
    SIGNALPR_LLM_MODEL=...  SIGNALPR_LLM_API_KEY=...  SIGNALPR_LLM_CACHE=use|refresh|only|off
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = Path(__file__).with_name("llm_models.yaml")
DEFAULT_CACHE = Path(os.environ.get("SIGNALPR_LLM_CACHE_DIR", ROOT / ".cache" / "llm"))
CACHE_MODES = ("use", "refresh", "only", "off")
# use: read + write · refresh: always call, overwrite · only: never call (replay; error on miss) · off: no cache


class LLMError(RuntimeError):
    pass


class CacheMiss(LLMError):
    pass


# ------------------------------------------------------------------ config
@dataclass
class LLMConfig:
    model: str
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_tokens: int = 1024
    timeout: float = 300.0
    max_retries: int = 5
    cache_mode: str = "use"
    cache_dir: str = str(DEFAULT_CACHE)
    preset: str | None = None
    backend: str | None = None

    def __post_init__(self):
        if self.cache_mode not in CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {self.cache_mode!r}")

    @classmethod
    def from_preset(cls, preset: str = "qwen-7b", backend: str = "ollama", **overrides) -> "LLMConfig":
        spec = _load_yaml(MODELS_YAML)
        try:
            model = spec["presets"][preset][backend]
        except KeyError:
            avail = {p: list(b) for p, b in spec["presets"].items()}
            raise ValueError(f"no preset {preset!r} for backend {backend!r}; available: {avail}") from None
        kw = {
            **spec.get("defaults", {}),
            **spec["backends"][backend],
            "model": model,
            "preset": preset,
            "backend": backend,
        }
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMConfig":
        d = _load_yaml(Path(path))
        d = d.get("llm", d)
        if "preset" in d:
            preset, backend = d.pop("preset"), d.pop("backend", "ollama")
            return cls.from_preset(preset, backend, **d)
        return cls(**d)

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        e = os.environ.get
        cfg = cls.from_preset(e("SIGNALPR_LLM_PRESET", "qwen-7b"), e("SIGNALPR_LLM_BACKEND", "ollama"))
        env = {
            "base_url": e("SIGNALPR_LLM_BASE_URL"),
            "model": e("SIGNALPR_LLM_MODEL"),
            "api_key": e("SIGNALPR_LLM_API_KEY"),
            "cache_mode": e("SIGNALPR_LLM_CACHE"),
        }
        return replace(cfg, **{k: v for k, v in env.items() if v}, **overrides)

    def settings(self) -> dict:
        """The generation settings that change the output (part of the cache key)."""
        return {"temperature": self.temperature, "top_p": self.top_p, "seed": self.seed, "max_tokens": self.max_tokens}


def _load_yaml(path: Path) -> dict:
    import yaml  # PyYAML (installed with huggingface_hub)

    return yaml.safe_load(path.read_text()) or {}


# ------------------------------------------------------------------ response
@dataclass
class LLMResponse:
    text: str
    model: str
    key: str
    cached: bool = False
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ cache
class PromptCache:
    """SQLite table of (key -> response). Safe across threads and processes."""

    def __init__(self, directory: str | Path):
        self.path = Path(directory) / "llm_cache.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY, model TEXT, prompt_hash TEXT, settings TEXT, sample INTEGER,
                tag TEXT, response TEXT, created REAL)""")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str) -> dict | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self, key: str, model: str, prompt_hash: str, settings: dict, sample: int, tag: str | None, response: dict
    ) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?)",
                (
                    key,
                    model,
                    prompt_hash,
                    json.dumps(settings, sort_keys=True),
                    sample,
                    tag,
                    json.dumps(response),
                    time.time(),
                ),
            )

    def stats(self) -> dict:
        with self._lock, self._conn() as c:
            n, models = c.execute("SELECT COUNT(*), COUNT(DISTINCT model) FROM responses").fetchone()
        return {"entries": n, "models": models, "path": str(self.path)}


def prompt_hash(messages: list[dict], json_mode: bool = False) -> str:
    blob = json.dumps({"messages": messages, "json_mode": json_mode}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_key(model: str, phash: str, settings: dict, sample: int) -> str:
    blob = json.dumps({"model": model, "prompt": phash, "settings": settings, "sample": sample}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ client
class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self.cache = PromptCache(self.config.cache_dir) if self.config.cache_mode != "off" else None
        self.calls = 0  # requests actually sent to the server
        self.hits = 0  # answered from cache

    # -- low level -------------------------------------------------------
    def _url(self, path: str) -> str:
        return self.config.base_url.rstrip("/") + path

    def _request(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._url(path),
            data=data,
            method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"},
        )
        last: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:  # noqa: S310 - configured URL
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code == 429 or e.code >= 500:
                    last = LLMError(f"HTTP {e.code}: {detail}")
                    wait = _retry_after(e.headers.get("Retry-After"), attempt)
                elif e.code == 400 and body is not None and "response_format" in body:
                    body = {k: v for k, v in body.items() if k != "response_format"}  # server lacks JSON mode
                    req = urllib.request.Request(
                        req.full_url,
                        data=json.dumps(body).encode("utf-8"),
                        method="POST",
                        headers=dict(req.header_items()),
                    )
                    continue
                else:
                    raise LLMError(f"HTTP {e.code} from {req.full_url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = LLMError(f"cannot reach {self._url('')}: {e}")
                wait = _retry_after(None, attempt)
            if attempt < self.config.max_retries:
                time.sleep(wait)
        raise last or LLMError("request failed")

    # -- public API ------------------------------------------------------
    def health(self) -> dict:
        """List the models the server is serving (raises LLMError if unreachable)."""
        data = self._request("/models")
        return {"ok": True, "models": [m.get("id") for m in data.get("data", [])], "base_url": self.config.base_url}

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        json_mode: bool = False,
        sample: int = 0,
        tag: str | None = None,
    ) -> LLMResponse:
        """One chat completion. `sample` > 0 asks for a *different* sample of the same prompt (seed + sample),
        which is cached under its own key; use it for self-consistency runs."""
        c = self.config
        settings = {
            "temperature": c.temperature if temperature is None else temperature,
            "top_p": c.top_p if top_p is None else top_p,
            "seed": (c.seed if seed is None else seed) + sample,
            "max_tokens": c.max_tokens if max_tokens is None else max_tokens,
        }
        phash = prompt_hash(messages, json_mode)
        key = cache_key(c.model, phash, settings, sample)
        if self.cache is not None and c.cache_mode in ("use", "only"):
            hit = self.cache.get(key)
            if hit is not None:
                self.hits += 1
                return LLMResponse(**{**hit, "cached": True, "latency_s": 0.0})
            if c.cache_mode == "only":
                raise CacheMiss(f"cache miss in replay mode (model={c.model}, prompt={phash[:12]}, sample={sample})")
        body = {"model": c.model, "messages": messages, **settings}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        data = self._request("/chat/completions", body)
        self.calls += 1
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        resp = LLMResponse(
            text=(choice.get("message") or {}).get("content") or "",
            model=c.model,
            key=key,
            latency_s=round(time.time() - t0, 3),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            settings=settings,
        )
        if self.cache is not None:
            self.cache.put(
                key,
                c.model,
                phash,
                settings,
                sample,
                tag,
                {k: v for k, v in resp.to_dict().items() if k not in ("cached", "latency_s")},
            )
        return resp

    def complete(self, prompt: str, system: str | None = None, **kw) -> LLMResponse:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)

    def chat_json(self, messages: list[dict], **kw):
        """Chat and parse the first JSON object/array in the reply (None if there is none)."""
        return extract_json(self.chat(messages, json_mode=True, **kw).text)

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.hits,
            "model": self.config.model,
            **({"cache": self.cache.stats()} if self.cache else {}),
        }


def _retry_after(header: str | None, attempt: int) -> float:
    try:
        return min(float(header), 60.0)
    except (TypeError, ValueError):
        return min(2.0**attempt, 30.0)


def extract_json(text: str):
    """First JSON object/array in an LLM reply; tolerates ```json fences, chatter and <think> blocks."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_c:
                    depth += 1
                elif text[i] == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except ValueError:
                            break
            start = text.find(open_c, start + 1)
    return None
