#!/usr/bin/env bash
# Run Qwen-Coder locally with Ollama (laptop / Kaggle / any GPU box) (issue #2).
#   bash scripts/serve/ollama_local.sh            # qwen2.5-coder:7b
#   MODEL=qwen2.5-coder:32b bash scripts/serve/ollama_local.sh
set -euo pipefail
MODEL="${MODEL:-qwen2.5-coder:7b}"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. macOS: brew install ollama   Linux: curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi
if ! curl -s -o /dev/null http://localhost:11434/api/tags; then
  echo "▶ starting ollama serve (log: /tmp/ollama.log)"
  (OLLAMA_KEEP_ALIVE=24h nohup ollama serve >/tmp/ollama.log 2>&1 &)
  for _ in $(seq 1 30); do curl -s -o /dev/null http://localhost:11434/api/tags && break; sleep 1; done
fi
echo "▶ pulling $MODEL (first time only)"
ollama pull "$MODEL"
echo "✓ ready: base_url=http://localhost:11434/v1 model=$MODEL"
echo "  python scripts/llm_throughput.py --backend ollama --model $MODEL"
