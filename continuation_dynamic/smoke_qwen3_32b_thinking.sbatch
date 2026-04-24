#!/bin/bash
#SBATCH --partition=ghx4
#SBATCH --account=bgye-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=smoke-qwen3_32b_thinking
#SBATCH --output=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/smoke_qwen3_32b_thinking_%j.out
#SBATCH --error=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/smoke_qwen3_32b_thinking_%j.err

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

echo "=== SMOKE qwen3_32b_thinking (Qwen/Qwen3-32B) at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU info"

srun python run_continuations.py \
    --model Qwen/Qwen3-32B \
    --data-dir /projects/bgye/yzhu38/narrative_project/NEWCORPUS_CLEANED \
    --results-dir /projects/bgye/yzhu38/narrative_project/continuation_dynamic/smoke_results/qwen3_32b_thinking \
    --no-id-filter --limit-stories 2 --thinking --max-tokens-mult 2.0

RC=$?
echo "=== SMOKE qwen3_32b_thinking exit code: $RC at $(date) ==="
exit $RC
