"""Measure LLM throughput to plan run budgets (issue #2).

Sends N review-sized requests (a ~250-line diff + review instructions, like the agent's specialist
prompts) with C in parallel, then reports latency, tokens/s and PRs/hour, and appends a row to
results/llm/throughput.csv.

    python scripts/llm_throughput.py --backend ollama --model qwen2.5-coder:7b
    python scripts/llm_throughput.py --preset qwen-32b --backend vllm --base-url http://localhost:8000/v1 \
        --n 40 --concurrency 8 --where sol-a100

PRs/hour assumes --calls-per-pr LLM calls per pull request: SignalPR makes about 10 (1 intent check +
3 specialists x 3 self-consistency runs); the single-prompt baseline makes 1.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.llm_client import LLMClient, LLMConfig  # noqa: E402

SYSTEM = (
    "You are a senior code reviewer. Review the diff for correctness, security and design problems. "
    'Reply with JSON only: {"comments": [{"file": "...", "line": 0, "severity": "critical|medium|low", '
    '"title": "...", "body": "...", "suggestion": "..."}]}. At most 6 comments.'
)

FUNCS = [
    'def load_user(db, user_id):\n    row = db.execute(f"SELECT * FROM users WHERE id = {user_id}").fetchone()\n'
    "    if row == None:\n        return {}\n    return dict(row)\n",
    "def retry(fn, attempts=3, delays=[1, 2, 4]):\n    for i in range(attempts):\n        try:\n"
    "            return fn()\n        except:\n            time.sleep(delays[i])\n",
    "def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += float(items[i]['price']) * "
    "items[i]['qty']\n    return round(s, 2)\n",
    "async def save(api, cart):\n    try:\n        api.post('/cart', json=cart)\n    except Exception:\n        pass\n",
    "def parse(cfg_text):\n    import yaml\n    return yaml.load(cfg_text)\n",
]


def make_diff(seed: int, lines: int = 250) -> str:
    rng = random.Random(seed)
    out, n = [f"# request {seed}: unique content so no response is reused\n"], 0
    for f in range(3):
        out.append(
            f"diff --git a/src/module_{seed}_{f}.py b/src/module_{seed}_{f}.py\n"
            f"--- a/src/module_{seed}_{f}.py\n+++ b/src/module_{seed}_{f}.py\n@@ -1,0 +1,{lines // 3} @@\n"
        )
        body = "".join(rng.choice(FUNCS) for _ in range(lines // 15))
        for ln in body.splitlines()[: lines // 3]:
            out.append("+" + ln + "\n")
            n += 1
    return "".join(out)


def one_call(llm: LLMClient, i: int, max_tokens: int) -> dict:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "PR: refactor data access helpers\n\nDIFF:\n" + make_diff(i)},
    ]
    t = time.time()
    try:
        r = llm.chat(msgs, max_tokens=max_tokens, json_mode=True, tag="throughput")
        return {"ok": True, "latency": time.time() - t, "pt": r.prompt_tokens or 0, "ct": r.completion_tokens or 0}
    except Exception as e:  # noqa: BLE001 - count failures, keep measuring
        return {"ok": False, "latency": time.time() - t, "pt": 0, "ct": 0, "error": str(e)[:200]}


def main(argv=None):
    p = argparse.ArgumentParser(description="LLM throughput benchmark (issue #2)")
    p.add_argument("--preset", default="qwen-7b")
    p.add_argument("--backend", default="ollama", choices=["ollama", "vllm"])
    p.add_argument("--model", default=None, help="override the preset's model name")
    p.add_argument("--base-url", default=None)
    p.add_argument("--n", type=int, default=20, help="number of requests")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--calls-per-pr", type=float, default=10, help="LLM calls per PR (SignalPR ~10, baseline 1)")
    p.add_argument("--where", default="unspecified", help="label for the hardware, e.g. sol-a100, kaggle-2xT4")
    p.add_argument("--out", default=str(ROOT / "results" / "llm" / "throughput.csv"))
    a = p.parse_args(argv)

    over = {"cache_mode": "off", "max_tokens": a.max_tokens, "temperature": 0.0}
    if a.model:
        over["model"] = a.model
    if a.base_url:
        over["base_url"] = a.base_url
    cfg = LLMConfig.from_preset(a.preset, a.backend, **over)
    llm = LLMClient(cfg)
    print(f"model={cfg.model} url={cfg.base_url} n={a.n} concurrency={a.concurrency}")
    llm.health()
    one_call(llm, -1, 16)  # warm-up: loads the model into GPU memory

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        res = list(ex.map(lambda i: one_call(llm, i, a.max_tokens), range(a.n)))
    wall = time.time() - t0

    ok = [r for r in res if r["ok"]]
    if not ok:
        sys.exit(f"all {a.n} requests failed; first error: {res[0].get('error')}")
    lat = sorted(r["latency"] for r in ok)
    calls_per_hour = len(ok) / wall * 3600
    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "where": a.where,
        "backend": a.backend,
        "model": cfg.model,
        "n": a.n,
        "ok": len(ok),
        "concurrency": a.concurrency,
        "max_tokens": a.max_tokens,
        "wall_s": round(wall, 1),
        "latency_p50_s": round(statistics.median(lat), 2),
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
        "prompt_tokens_mean": round(statistics.mean(r["pt"] for r in ok)),
        "completion_tokens_mean": round(statistics.mean(r["ct"] for r in ok)),
        "output_tok_per_s": round(sum(r["ct"] for r in ok) / wall, 1),
        "calls_per_hour": round(calls_per_hour),
        "calls_per_pr": a.calls_per_pr,
        "prs_per_hour": round(calls_per_hour / a.calls_per_pr, 1),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

    # *.csv is git-ignored in this repo, so also keep a Markdown log that can be committed
    md = out.with_suffix(".md")
    cols = [
        "date",
        "where",
        "model",
        "concurrency",
        "latency_p50_s",
        "output_tok_per_s",
        "calls_per_hour",
        "prs_per_hour",
    ]
    if not md.exists():
        md.write_text(
            "# LLM throughput (scripts/llm_throughput.py)\n\n| "
            + " | ".join(cols)
            + " |\n|"
            + "---|" * len(cols)
            + "\n"
        )
    with md.open("a") as f:
        f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")

    width = max(len(k) for k in row)
    for k, v in row.items():
        print(f"{k:>{width}}  {v}")
    pph = row["prs_per_hour"]
    print("\nRun budget at this speed (1 repeat):")
    for name, prs in [("Martian (dev)", 50), ("ReviewBench (test)", 219), ("c-CRAB", 184)]:
        print(f"  {name:20s} {prs:4d} PRs  ≈ {prs / pph:5.1f} h   (x3 repeats ≈ {3 * prs / pph:5.1f} h)")
    if len(ok) < a.n:
        print(f"\n{a.n - len(ok)} request(s) failed, e.g.: {next(r['error'] for r in res if not r['ok'])}")
    print(f"\nappended to {out} and {md}")


if __name__ == "__main__":
    main()
