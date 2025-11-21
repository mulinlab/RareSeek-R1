#!/bin/bash
#SBATCH --job-name=RareSeek-R1
#SBATCH --nodes=1
#SBATCH --nodelist=gpu3
#SBATCH -p llm
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export LOGLEVEL=INFO

echo "=== Job started at $(date) on $(hostname) ==="

vllm serve TaoMedAI/model/RareSeek-R1 \
	--host 0.0.0.0 \
	--port 8001 \
	--max_model_len 2048 \
    --tensor-parallel-size 4 \
	--gpu_memory_utilization 0.93
