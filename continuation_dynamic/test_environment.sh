#!/bin/bash
#SBATCH --partition=ghx4
#SBATCH --account=bgye-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=env-test
#SBATCH --output=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/test_env_%j.out
#SBATCH --error=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/test_env_%j.err

set -e

echo "=========================================="
echo "ENVIRONMENT VALIDATION TEST"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# 1. GPU Check
echo ""
echo "=== 1. GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 2. Environment
echo ""
echo "=== 2. Module and Conda ==="
module load python/miniforge3_pytorch
conda activate /projects/bgye/envs/llm
echo "Python: $(which python)"
python --version

# 3. Package Versions
echo ""
echo "=== 3. Package Versions ==="
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
python -c "import transformers; print('Transformers:', transformers.__version__)"

# 4. HF Auth
echo ""
echo "=== 4. HF Authentication ==="
export HF_HOME=/projects/bgye/models/hf_cache
# Load HF_TOKEN (and any other secrets) from project .env (gitignored)
set -a; source /projects/bgye/yzhu38/narrative_project/.env; set +a
python -c "from huggingface_hub import login; login(token='${HF_TOKEN:?HF_TOKEN not set; source .env first}', add_to_git_credential=False); print('HF login OK')"

# 5. Run full Python validation
echo ""
echo "=== 5. Running Python validation ==="
python /projects/bgye/yzhu38/narrative_project/continuation_dynamic/validate_models.py

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETED: $(date)"
echo "=========================================="