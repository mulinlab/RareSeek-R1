#!/bin/bash
#SBATCH --job-name=RareSeek-R1
#SBATCH --nodes=1
#SBATCH --nodelist=gpu3
#SBATCH -p llm
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT="$SLURM_SUBMIT_DIR"
    if [[ ! -f "$ROOT/inference.py" ]]; then
        ROOT="$(cd -- "$ROOT/.." && pwd)"
    fi
else
    ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -f "$ROOT/inference.py" ]]; then
    echo "Submit this job from the repository root or its inference directory." >&2
    exit 1
fi
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export LOGLEVEL=INFO

echo "=== Job started at $(date) on $(hostname) ==="

vllm serve models/RareSeek-R1 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95
