#!/bin/bash
#SBATCH --partition=ghx4
#SBATCH --account=bgye-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=cont-llama4_scout_base
#SBATCH --output=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/cont_llama4_scout_base_%j.out
#SBATCH --error=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/cont_llama4_scout_base_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bgye/yzhu38/cache/torchinductor
export TRITON_CACHE_DIR=/work/nvme/bgye/yzhu38/cache/triton
export HF_HOME=/projects/bgye/models/hf_cache
# Load HF_TOKEN (and any other secrets) from project .env (gitignored)
set -a; source /projects/bgye/yzhu38/narrative_project/.env; set +a
export VLLM_CACHE_ROOT=/work/nvme/bgye/yzhu38/cache/vllm-$SLURM_JOB_ID
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $VLLM_CACHE_ROOT

module load python/miniforge3_pytorch
conda activate /projects/bgye/envs/llm

cd /projects/bgye/yzhu38/narrative_project/continuation_dynamic

echo "=== meta-llama/Llama-4-Scout-17B-16E continuations at $(date) ==="
echo "Short name: llama4_scout_base"
echo "Node: $(hostname), GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',' || echo 'N/A')"

srun python run_continuations.py \
    --model meta-llama/Llama-4-Scout-17B-16E \
    --data-dir /projects/bgye/yzhu38/narrative_project/NEWCORPUS_CLEANED \
    --results-dir /projects/bgye/yzhu38/narrative_project/continuation_dynamic/results_llama4_scout_base \
    --no-id-filter --tensor-parallel-size 2

echo "=== Done at $(date) ==="
