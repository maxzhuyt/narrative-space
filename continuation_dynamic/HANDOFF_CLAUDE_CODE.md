# Handoff Document — Claude Code Takeover

## What This Project Is

Rewrite the continuation generation pipeline on the DeltaAI (NCSA) cluster to run a multi-model narrative predictability study comparing base vs instruct vs reasoning models.

## Cluster & Environment

- **Cluster**: DeltaAI (NCSA), partition `ghx4`
- **Account**: `bgye-dtai-gh`
- **Nodes**: 4× NVIDIA GH200 120GB (Grace-Hopper Superchip) per node
- **Module**: `python/miniforge3_pytorch`
- **Conda env**: `/projects/bgye/envs/llm` (vLLM 0.19.1, PyTorch 2.10.0+cu129, transformers 5.6.2)
- **GPU allocation**: `--gpus-per-node=1` for most models, `--gpus-per-node=4` for Llama-4-Scout (TP=4 — 218 GB bf16 weights need 4×96 GiB HBM3; Slurm `gh200_120gb` label is CPU-side LPDDR5X, not HBM)
- **Project path**: `/projects/bgye/yzhu38/narrative_project/`
- **NVMe cache**: `/work/nvme/bgye/yzhu38/cache/`
- **HF cache**: `/projects/bgye/models/hf_cache/`
- **HF token**: `<REDACTED — set HF_TOKEN via project .env (gitignored)>`
- **Node.js**: Installed at `/u/yzhu38/.local/` (needed for MCP servers)
- **opencode config**: `/u/yzhu38/.config/opencode/opencode.json`

## Comparison Groups (Finalized)

### Dense — Base vs Instruct (same arch, different training)

| # | Base | Instruct | Params | Notes |
|---|---|---|---|---|
| 1 | `Qwen/Qwen2.5-32B` | `Qwen/Qwen2.5-32B-Instruct` | 32B dense | TP=1 |
| 2 | `mistralai/Mistral-Small-24B-Base-2501` | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B dense | TP=1, needs `tokenizer_mode=mistral` |
| 3 | `google/gemma-4-31B` | `google/gemma-4-31B-it` | 33B dense | TP=1, needs `limit_mm_per_prompt={"image": 0}` |
| 4 | `meta-llama/Llama-4-Scout-17B-16E` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 109B MoE (17B active) | TP=4, needs `limit_mm_per_prompt={"image": 0}`, **gated model** |

### Dense — Instruct vs Reasoning (same family, thinking on/off)

| # | Non-thinking | Thinking | Notes |
|---|---|---|---|
| 5 | `Qwen/Qwen3-32B` (thinking=False) | `Qwen/Qwen3-32B` (thinking=True) | 32B dense, `--thinking` flag |
| 6 | `Qwen/Qwen3.5-35B-A3B` (thinking=False) | `Qwen/Qwen3.5-35B-A3B` (thinking=True) | 35B MoE (3B active) |

### Dense — Reasoning only

| # | Model | Notes |
|---|---|---|
| 7 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Always produces thinking tokens, user-only prompt |

**Total: 13 runs** (some models run twice with/without `--thinking`).

## Key Design Decisions

1. **Base models get raw text only** — `BASE_PROMPT_TEMPLATE = "{story_so_far}"`. No instruction, no "CONTINUATION:" label. Pure next-token prediction.
2. **Instruct/reasoning models** get a system prompt + user prompt via chat template.
3. **Reasoning models with `--thinking`**: `enable_thinking=True` passed to `apply_chat_template`, output stripped of `` blocks.
4. **Reasoning models without `--thinking`**: `enable_thinking=False` (default), generates in instruct mode, no stripping needed.
5. **DeepSeek-R1** always uses user-only prompt (no system message per official docs), always thinks, always strips.
6. **Same generation params** across all models: `temperature=1.2`, `top_p=0.95`, `n=5` continuations per position.
7. **Reasoning models** get `--max-tokens-mult 2.0` to accommodate thinking tokens.
8. **Qwen3-32B runs twice**: once with `--thinking` (results_qwen3_32b_thinking) and once without (results_qwen3_32b).
9. **Qwen3.5-35B-A3B runs twice**: same pattern.
10. **Llama-4-Scout needs TP=4** (109B params in bf16 ≈ 218GB, needs all 4 GPUs on one node).
11. **Llama-4-Scout is gated** — user needs to accept Meta's license on HuggingFace.
12. **Llama-4-Scout is vision-language** — needs `limit_mm_per_prompt={"image": 0}` for text-only.
13. **Gemma-4-31B is vision-language** — same `limit_mm_per_prompt` setting.
14. **Mistral-Small** needs `tokenizer_mode=mistral`, `config_format=mistral`, `load_format=mistral`.
15. **Old Qwen3-32B results** (20 continuations) already moved to `results_qwen3_32b/`.

## File Structure

```
/projects/bgye/yzhu38/narrative_project/continuation_dynamic/
├── run_continuations.py      # Main script — CORRUPTED, needs full rewrite
├── validate_models.py         # GPU/vLLM/model validation script
├── test_environment.sh        # sbatch wrapper for validation
├── submit_all.sh              # Batch submit all jobs — needs update
├── run_*.sbatch               # Old sbatch files — need replacement
├── logs/                      # SLURM log output directory
└── results_qwen3_32b/        # Old results (20 continuations, kept)
```

## What Needs to Be Done

### 1. Rewrite `run_continuations.py` (CRITICAL — currently corrupted/empty)

The file was accidentally zeroed out during editing. It needs a complete rewrite with these changes from the previous version:

- **`BASE_PROMPT_TEMPLATE`** changed to `"{story_so_far}"` (raw story prefix only, no instruction)
- **Added `--thinking` flag** (`action="store_true"`) — controls `enable_thinking` in chat template and `strip_thinking` logic
- **Added `--tensor-parallel-size`** arg (default 1, use 4 for Llama-4-Scout)
- **Added `--quantization`** arg (default None, for future FP8 use)
- **`get_llm_kwargs()`** now accepts `tensor_parallel_size` and `quantization` params
- **`enable_thinking`** now uses `args.thinking` instead of hardcoded `False`
- **`strip_think`** = `is_reasoning_model(model_id) and args.thinking` (not just `is_reasoning_model`)
- **Results JSON** now includes `thinking_enabled`, `tensor_parallel_size`, `quantization` fields
- **Model classification sets** updated (see below)
- **Print statements** added for `thinking`, `tensor_parallel_size`, `quantization`, prompt mode, strip_think, is_deepseek_r1

### 2. Model Classification Sets

```python
BASE_MODELS = {
    "Qwen/Qwen2.5-32B",
    "mistralai/Mistral-Small-24B-Base-2501",
    "google/gemma-4-31B",
    "meta-llama/Llama-4-Scout-17B-16E",
    "Qwen/Qwen3.5-35B-A3B-Base",
}

REASONING_MODELS = {
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-35B-A3B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

VLM_MODELS = {
    "google/gemma-4-31B",
    "google/gemma-4-31B-it",
    "meta-llama/Llama-4-Scout-17B-16E",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}

MISTRAL_MODELS = {
    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mistral-Small-24B-Instruct-2501",
}
```

### 3. Prompt Definitions

```python
SYSTEM_PROMPT = (
    "You are a fiction writer. Continue the story naturally in the "
    "same style and voice. Write only story text \u2014 no commentary, "
    "no meta-discussion, no preamble, no quotation marks around "
    "your continuation."
)

COMPLETION_TEMPLATE = (
    "Continue this story to its conclusion in approximately {n_words} words. "
    "Maintain the same tone, style, and narrative voice throughout. "
    "Do not summarize or describe what happens \u2014 write the actual story "
    "text as it would appear on the page.\n\n"
    "STORY SO FAR:\n{story_so_far}"
)

BASE_PROMPT_TEMPLATE = "{story_so_far}"

DEEPSEEK_USER_TEMPLATE = (
    "Continue this story to its conclusion in approximately {n_words} words. "
    "Maintain the same tone, style, and narrative voice throughout. "
    "Do not summarize or describe what happens \u2014 write the actual story "
    "text as it would appear on the page. "
    "Please reason step by step, and write the story continuation directly.\n\n"
    "STORY SO FAR:\n{story_so_far}"
)
```

### 4. Create 13 New sbatch Files

| # | File | Model | Special flags | GPUs |
|---|---|---|---|---|
| 1 | `run_qwen25_32b.sbatch` | `Qwen/Qwen2.5-32B` | base model | 1 |
| 2 | `run_qwen25_32b_instruct.sbatch` | `Qwen/Qwen2.5-32B-Instruct` | — | 1 |
| 3 | `run_mistral_small_24b_base.sbatch` | `mistralai/Mistral-Small-24B-Base-2501` | base model | 1 |
| 4 | `run_mistral_small_24b_instruct.sbatch` | `mistralai/Mistral-Small-24B-Instruct-2501` | — | 1 |
| 5 | `run_gemma4_31b.sbatch` | `google/gemma-4-31B` | base model, VLM | 1 |
| 6 | `run_gemma4_31b_it.sbatch` | `google/gemma-4-31B-it` | VLM | 1 |
| 7 | `run_llama4_scout_base.sbatch` | `meta-llama/Llama-4-Scout-17B-16E` | base, VLM, `--tensor-parallel-size 4` | 4 |
| 8 | `run_llama4_scout_instruct.sbatch` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | VLM, `--tensor-parallel-size 4` | 4 |
| 9 | `run_qwen3_32b.sbatch` | `Qwen/Qwen3-32B` | reasoning, thinking=False | 1 |
| 10 | `run_qwen3_32b_thinking.sbatch` | `Qwen/Qwen3-32B` | reasoning, `--thinking`, `--max-tokens-mult 2.0` | 1 |
| 11 | `run_qwen35_35b_a3b.sbatch` | `Qwen/Qwen3.5-35B-A3B` | reasoning, thinking=False | 1 |
| 12 | `run_qwen35_35b_a3b_thinking.sbatch` | `Qwen/Qwen3.5-35B-A3B` | reasoning, `--thinking`, `--max-tokens-mult 2.0` | 1 |
| 13 | `run_deepseek_r1_distill_qwen_32b.sbatch` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | reasoning, always thinks, `--max-tokens-mult 2.0` | 1 |

### 5. sbatch Template (1-GPU models)

```bash
#!/bin/bash
#SBATCH --partition=ghx4
#SBATCH --account=bgye-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=cont-<SHORTNAME>
#SBATCH --output=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/cont_<SHORTNAME>_%j.out
#SBATCH --error=/projects/bgye/yzhu38/narrative_project/continuation_dynamic/logs/cont_<SHORTNAME>_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bgye/yzhu38/cache/torchinductor
export TRITON_CACHE_DIR=/work/nvme/bgye/yzhu38/cache/triton
export HF_HOME=/projects/bgye/models/hf_cache
export HF_TOKEN=<REDACTED — set HF_TOKEN via project .env (gitignored)>
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load python/miniforge3_pytorch
conda activate /projects/bgye/envs/llm

cd /projects/bgye/yzhu38/narrative_project/continuation_dynamic

echo "=== <MODEL_ID> continuations at $(date) ==="
echo "Model type: <base|instruct|reasoning>"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

srun python run_continuations.py \
    --model <MODEL_ID> \
    --data-dir /projects/bgye/yzhu38/narrative_project/NEWCORPUS_CLEANED \
    --results-dir /projects/bgye/yzhu38/narrative_project/continuation_dynamic/results_<SHORTNAME> \
    --no-id-filter <EXTRA_FLAGS>

echo "=== Done at $(date) ==="
```

For Llama-4-Scout models (TP=4), change: `--gpus-per-node=4` and add `--tensor-parallel-size 4 --gpu-util 0.92` to the python command.

For reasoning models with thinking: add `--thinking --max-tokens-mult 2.0`.
For reasoning models without thinking: add `--max-tokens-mult 2.0` (DeepSeek always uses this).

### 6. Delete Old sbatch Files

These old files should be removed and replaced:

- `run_continuations.sbatch`
- `run_shard[0-5].sbatch`
- `run_qwen3_32b.sbatch`
- `run_mistral_small_24b_base.sbatch`
- `run_qwen3_30b_a3b.sbatch`
- `run_qwen35_35b_a3b.sbatch`
- `run_gemma4_31b.sbatch`
- `run_deepseek_r1_distill_qwen_32b.sbatch`
- `run_mistral_small_24b_instruct.sbatch`
- `run_qwen25_32b.sbatch`
- `run_gemma4_31b_it.sbatch`
- `run_qwen25_32b_instruct.sbatch`

### 7. Update `submit_all.sh`

Replace with the new 13 sbatch files.

### 8. Validate and Submit

1. Verify Llama-4-Scout access — HF token needs to have accepted Meta's license agreement for both base and instruct variants
2. Validate environment on compute node (`salloc` + `test_environment.sh`)
3. Submit all jobs via `submit_all.sh`

## Risks & Unknowns

- **Llama-4-Scout is gated**: User may need to go to HuggingFace and accept Meta's license for both base and instruct variants before the HF token can download them
- **vLLM 0.19.1 support**: Llama-4-Scout support requires vLLM ≥ 0.8.3; vLLM 0.19.1 should work but hasn't been tested on this cluster
- **Gemma-4-31B**: Vision-language model, needs `limit_mm_per_prompt={"image": 0}` — also needs verification on vLLM 0.19.1
- **Qwen3.5-35B-A3B**: MoE model (35B total, 3B active), loads all params (~70GB bf16), should fit on 1 GPU
- **salloc was previously revoked** — may need to try again or go straight to `sbatch` submission
- **Base model prompt philosophy**: Base models should receive ONLY the raw story prefix text (`"{story_so_far}"`), with NO instructional framing, NO "CONTINUATION:" label, and NO word count guidance. This measures pure next-token prediction without instruction compliance confounds.
- **Thinking mode logic**: `strip_think` should only be True when BOTH `is_reasoning_model(model_id)` AND `args.thinking` are True. If `--thinking` is not set, reasoning models generate in instruct mode (no thinking tokens), so there's nothing to strip.
- **DeepSeek-R1 is special**: It always thinks, always uses user-only prompt, always strips. It does NOT use `enable_thinking` (that's a Qwen feature).

## MCP Servers Status

Node.js was installed at `/u/yzhu38/.local/` because it was missing from the cluster. The opencode config at `/u/yzhu38/.config/opencode/opencode.json` was updated to use absolute paths for `npx`. MCP servers (brave-search, context7, playwright) should work after restarting opencode.

## Old Results

The previous run with Qwen/Qwen3-32B using 20 continuations has been moved to `results_qwen3_32b/`. The new runs use `N_CONTINUATIONS = 5`.
