# Serving Qwen-Coder + the shared LLM client (issue #2)

Every LLM call in the project (agent #11, baselines #8/#9) goes through `agent/llm_client.py`.
It talks to any OpenAI-compatible server, so pick whichever place has a free GPU:

| Where | Server | 7B | 32B | Start with |
|---|---|---|---|---|
| ASU Sol | vLLM | A100 | A100 80 GB (bf16) | `sbatch scripts/serve/sol_vllm.sbatch` |
| Kaggle (free, 30 GPU h/week) | Ollama | 2× T4 | 2× T4 (4-bit) | `notebooks/serve_qwen_kaggle.ipynb` |
| Laptop | Ollama | Apple Silicon / 16 GB RAM | needs ~24 GB RAM | `bash scripts/serve/ollama_local.sh` |

## Use the client

```python
from agent.llm_client import LLMClient, LLMConfig

llm = LLMClient(LLMConfig.from_preset("qwen-7b", backend="ollama"))  # or "qwen-32b", backend="vllm"
r = llm.chat([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
print(r.text, r.cached, r.latency_s, r.prompt_tokens, r.completion_tokens)

data = llm.chat_json(messages)  # parsed JSON reply
runs = [llm.chat(messages, temperature=0.7, sample=k) for k in range(3)]  # self-consistency samples
```

- **Config:** `model`, `temperature`, `top_p`, `seed`, `max_tokens`, `timeout`, `max_retries`, `base_url`.
  Presets live in `agent/llm_models.yaml`. You can also load a run config with
  `LLMConfig.from_yaml("configs/run.yaml")` (keys under `llm:`), or set environment variables
  `SIGNALPR_LLM_PRESET`, `SIGNALPR_LLM_BACKEND`, `SIGNALPR_LLM_BASE_URL`, `SIGNALPR_LLM_MODEL`.
- **Cache:** every prompt/response pair is stored in `.cache/llm/llm_cache.sqlite3` (git-ignored), keyed by
  **model + prompt hash + settings + sample index**. Re-running an experiment is free and gives identical
  outputs. `sample=k` gives the k-th independent sample (seed + k), each cached separately, so
  self-consistency runs don't collapse into one cached answer.
  Modes (`SIGNALPR_LLM_CACHE` or `cache_mode=`): `use` (default) · `refresh` (always call, overwrite) ·
  `only` (replay a finished run; error on a miss) · `off`.
  To share results, copy the `.sqlite3` file: teammates can then replay a run without a GPU.
- **Reliability:** retries on HTTP 429/5xx and connection errors, with back-off. If a server rejects JSON mode
  (400), the client drops it and retries.

## ASU Sol (vLLM)

One-time setup on a Sol login node (connect to the ASU VPN first if you're off campus):

```bash
ssh <asurite>@login.sol.rc.asu.edu
module load mamba/latest
mamba create -n vllm python=3.11 -y
source activate vllm
pip install vllm
git clone https://github.com/pranjalpadakannaya/SignalPR.git && cd SignalPR
```

Start a server (weights download to `/scratch/$USER/hf` the first time: ~15 GB for 7B, ~65 GB for 32B):

```bash
mkdir -p logs
sbatch scripts/serve/sol_vllm.sbatch                                          # 7B
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct sbatch scripts/serve/sol_vllm.sbatch    # 32B
squeue -u $USER                       # wait for state R
cat logs/vllm_endpoint.txt            # e.g. node=sg041 port=8000 model=...
tail -f logs/vllm.<jobid>.err         # ready when you see "Application startup complete"
```

Use it:
- **From Sol itself** (another job, or the login node): `SIGNALPR_LLM_BACKEND=vllm SIGNALPR_LLM_BASE_URL=http://<node>:8000/v1`
- **From your laptop**, through an SSH tunnel:
  `ssh -N -L 8000:<node>:8000 <asurite>@login.sol.rc.asu.edu`, then `--base-url http://localhost:8000/v1`

The job runs for up to 8 hours (`-t 0-08:00`). `scancel <jobid>` when done so the GPU is free for others.

## Kaggle (Ollama)

Open `notebooks/serve_qwen_kaggle.ipynb` in Kaggle → *Settings → Accelerator: GPU T4 x2, Internet: On* →
run all cells. The notebook installs Ollama, pulls the model, smoke-tests the client and runs the
throughput benchmark. Its last cell shows how to reach the server from your laptop with a temporary
Cloudflare tunnel.

## Laptop (Ollama)

```bash
brew install ollama            # macOS (Linux: curl -fsSL https://ollama.com/install.sh | sh)
bash scripts/serve/ollama_local.sh                      # 7B
MODEL=qwen2.5-coder:32b bash scripts/serve/ollama_local.sh
```

## Throughput (PRs/hour) for run budgets

```bash
python scripts/llm_throughput.py --backend ollama --model qwen2.5-coder:7b --where macbook
python scripts/llm_throughput.py --preset qwen-32b --backend vllm --base-url http://localhost:8000/v1 \
    --n 40 --concurrency 8 --where sol-a100
```

The script sends review-sized requests (~250-line diff) in parallel and prints latency, output tokens/s,
calls/hour and **PRs/hour**. `--calls-per-pr 10` matches SignalPR (1 intent + 3 specialists × 3 runs); use 1
for the single-prompt baseline. It also prints estimated hours for Martian, ReviewBench and c-CRAB, and
appends a row to `results/llm/throughput.md`, which is committed. The CSV next to it is git-ignored.

| Where | Model | PRs/hour | Notes |
|---|---|---|---|
| Sol A100 80 GB | 7B | _to measure_ | |
| Sol A100 80 GB | 32B | _to measure_ | |
| Kaggle 2× T4 | 7B | _to measure_ | |
| Kaggle 2× T4 | 32B (4-bit) | _to measure_ | |
