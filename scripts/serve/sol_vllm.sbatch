#!/bin/bash
# Serve a Qwen-Coder model with vLLM on one ASU Sol A100 (issue #2).
#   mkdir -p logs
#   sbatch scripts/serve/sol_vllm.sbatch                                          # 7B (default)
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct sbatch scripts/serve/sol_vllm.sbatch    # 32B (bf16 fits one A100 80GB)
# The node and port are written to logs/vllm_endpoint.txt; connect with scripts/serve/README.md § Sol.
#SBATCH -J signalpr-vllm
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-08:00
#SBATCH -o logs/vllm.%j.out
#SBATCH -e logs/vllm.%j.err

set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"

module load mamba/latest
source activate "${VLLM_ENV:-vllm}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"        # model weights go to scratch, not home (quota)
mkdir -p "$HF_HOME" logs

echo "node=$(hostname) port=$PORT model=$MODEL job=$SLURM_JOB_ID started=$(date -Is)" | tee logs/vllm_endpoint.txt
nvidia-smi --query-gpu=name,memory.total --format=csv

vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.92 \
  --seed 0 \
  --disable-log-requests
