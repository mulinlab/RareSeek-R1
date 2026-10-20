#!/usr/bin/env bash
#SBATCH --job-name=RareSeek-R1
#SBATCH --nodes=1
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
python "$ROOT/inference.py" --tensor-parallel-size 4 "$@"
