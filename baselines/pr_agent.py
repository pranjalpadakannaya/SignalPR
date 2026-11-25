"""PR-Agent baseline (issue #9): the open-source Qodo PR-Agent `review` command, run offline.

PR-Agent is installed in its own virtualenv (it needs Python >= 3.12 and pins its own litellm), and called
as a subprocess in its "plain diff" mode: the PR's unified diff goes in, a structured JSON review comes
out, with no GitHub access. The model is the same Qwen server the other systems use, so scores compare
within one model. Its `key_issues_to_review` become our common `Comment` format.

Setup: see baselines/README.md. Binary: $SIGNALPR_PR_AGENT_BIN (default `pr-agent` on PATH).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from agent.llm_client import LLMConfig
from agent.state import SEVERITIES, Comment, PRInput
from tools.diff import to_unified_diff

CACHE_DIR = Path(os.environ.get("SIGNALPR_PR_AGENT_CACHE", ".cache/pr_agent"))
TIMEOUT_S = int(os.environ.get("SIGNALPR_PR_AGENT_TIMEOUT", "900"))
# PR-Agent labels each key issue with a free-text header ("Possible Bug", "Security concern", ...).
SEVERITY_BY_HEADER = (("security", "critical"), ("bug", "medium"), ("error", "medium"), ("race", "medium"))


class PRAgentError(RuntimeError):
    pass


def binary() -> str:
    exe = os.environ.get("SIGNALPR_PR_AGENT_BIN") or shutil.which("pr-agent")
    if not exe:
        raise PRAgentError("pr-agent not found: install it (baselines/README.md) or set SIGNALPR_PR_AGENT_BIN")
    return exe


def litellm_model(model: str) -> str:
    """PR-Agent talks to Ollama through litellm: `ollama/<tag>`. vLLM/OpenAI-compatible servers use `openai/<id>`."""
    return model if "/" in model and model.split("/")[0] in ("ollama", "openai") else f"ollama/{model}"


def env_for(cfg: LLMConfig) -> dict[str, str]:
    """Point PR-Agent at the same model server as the rest of the project (Dynaconf env-var syntax)."""
    base = cfg.base_url.rstrip("/")
    env = {
        "CONFIG__CUSTOM_MODEL_MAX_TOKENS": os.environ.get("SIGNALPR_PR_AGENT_CTX", "16000"),
        "CONFIG__AI_TIMEOUT": str(TIMEOUT_S),
        "CONFIG__TEMPERATURE": str(cfg.temperature),
        "CONFIG__SEED": str(cfg.seed),
    }
    if cfg.backend != "ollama" or ":11434" not in base and "ollama" not in base:  # OpenAI-compatible server (vLLM on Sol, tunnel, ...)
        model = f"openai/{cfg.model}"
        env |= {"OPENAI__KEY": cfg.api_key or "EMPTY", "OPENAI__API_BASE": base}
    else:  # Ollama: litellm wants the host without /v1
        model = litellm_model(cfg.model)
        env["OLLAMA__API_BASE"] = base.removesuffix("/v1")
    env["CONFIG__MODEL"] = model
    env["CONFIG__FALLBACK_MODELS"] = json.dumps([model])
    return env


def _clean(v) -> str:
    return str(v or "").strip()


def severity_for(header: str) -> str:
    h = header.lower()
    for key, sev in SEVERITY_BY_HEADER:
        if key in h:
            return sev
    return "medium"


def parse_review(review: dict, pr: PRInput) -> list[Comment]:
    """Convert PR-Agent's JSON review into Comments. Issues with no file are dropped (can't be located)."""
    out: list[Comment] = []
    issues = (review.get("review") or review).get("key_issues_to_review") or []
    if isinstance(issues, str):  # some model replies put a bare string here
        return out
    for it in issues:
        if not isinstance(it, dict) or not _clean(it.get("relevant_file")):
            continue
        path = _clean(it["relevant_file"]).lstrip("/")
        f = pr.file(path)
        try:
            line = int(it.get("start_line"))
        except (TypeError, ValueError):
            line = None
        header = _clean(it.get("issue_header")) or "Issue"
        sev = severity_for(header)
        out.append(
            Comment(
                file=f.path if f else path,
                line=line,
                category="security" if "security" in header.lower() else "correctness",
                severity=sev if sev in SEVERITIES else "medium",
                title=header[:140],
                body=_clean(it.get("issue_content"))[:1500],
                source="baseline/pr_agent",
                quote=_clean(it.get("existing_code"))[:300],
                suggestion=_clean(it.get("improved_code"))[:500],
                support=1.0,
                snippet=f.snippet(line, 2) if f and line else [],
            )
        )
    return out


def run_pr_agent(diff: str, cfg: LLMConfig) -> dict:
    """Run `pr-agent review` on a diff and return its JSON output (cached on disk by diff + model + settings)."""
    env = env_for(cfg)
    key = hashlib.sha256(json.dumps([diff, sorted(env.items())]).encode()).hexdigest()[:32]
    cached = CACHE_DIR / f"{key}.json"
    if cfg.cache_mode in ("use", "only") and cached.exists():
        return json.loads(cached.read_text())
    if cfg.cache_mode == "only":
        raise PRAgentError("cache miss in replay mode")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "review.json"
        # cwd is an empty temp dir so PR-Agent finds no repo to read: it reviews the patch only.
        proc = subprocess.run(
            [binary(), "--stdin", "--json-output", str(out), "review"],
            input=diff, text=True, capture_output=True, cwd=tmp, timeout=TIMEOUT_S + 60,
            env={**os.environ, **env},
        )
        if not out.exists():
            raise PRAgentError(f"pr-agent produced no review (exit {proc.returncode}): {(proc.stderr or proc.stdout)[-300:]}")
        data = json.loads(out.read_text())
    if cfg.cache_mode != "off":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data))
    return data


def pr_agent_comments(pr: PRInput, cfg: LLMConfig) -> list[Comment]:
    return parse_review(run_pr_agent(to_unified_diff(pr.files), cfg), pr)
