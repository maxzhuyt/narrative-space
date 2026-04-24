# Narrative Predictability — Multi-Model Continuation Study

Measuring how language models predict story continuations as context is progressively revealed. Unpredictability (the model's persistent inability to anticipate the rest of the story) is used as a proxy for narrative surprise.

**Current experiment**: a 20-run comparison across base, instruct, and reasoning models on a shared 5001-story corpus, generating 5 continuations per position at 4 revealed-fraction positions (40/60/80/90%). Includes the full Olmo-3 32B post-training staircase (base → SFT → DPO → RLVR for both Instruct and Think branches) for a within-family stage-by-stage ablation.

> If you're looking for the earlier v1–v4 close-reading / 100-ending-distance work, see [`archive/2026-04-cleanup/README_pre_continuation_dynamic.md`](archive/2026-04-cleanup/README_pre_continuation_dynamic.md). All the old code lives in `archive/2026-04-cleanup/`.

---

## Directory layout

```
narrative_project/
├── README.md                        ← you are here
├── requirements.txt                 ← Python deps (reference; real env is conda)
├── NEWCORPUS_CLEANED/               ← 5001 .txt stories — input corpus
├── continuation_dynamic/            ← current generation pipeline (see below)
└── archive/                         ← superseded material
    ├── (16 legacy scripts from v1–v4 work)
    └── 2026-04-cleanup/             ← April 2026 cleanup snapshot
        ├── README_pre_continuation_dynamic.md   ← old top-level README
        ├── old_scripts/             ← previous top-level .py / .sbatch
        ├── old_dirs/                ← narrative_ablation, texts, distances, figures, logs, cache, __pycache__, GUTENBERG_CHILDREN
        ├── old_notebooks/           ← analysis_*.ipynb, sample_stories.ipynb
        └── old_orphans/             ← loose JSON / CSV / PNG / MD / TeX
```

---

## The current pipeline (`continuation_dynamic/`)

### Script: [`run_continuations.py`](continuation_dynamic/run_continuations.py)

Reads stories from `NEWCORPUS_CLEANED/` (or any `--data-dir`), splits each into sentences, selects 4 revealed-fraction cutpoints (40/60/80/90%), and asks a vLLM-hosted model to generate 5 continuations per cutpoint. Each continuation's target length equals the remaining ground-truth text (measured in words, then converted to a dynamic `max_tokens` budget). Output is one JSON per story with prefix, ground-truth continuation, and the 5 generated alternatives at each position.

Key flags (see `--help` for all):

| Flag | Purpose |
|---|---|
| `--model` | HF model id, e.g. `Qwen/Qwen2.5-32B` |
| `--data-dir` | Input corpus directory (default: `NEWCORPUS_CLEANED`) |
| `--results-dir` | Per-model output directory (one JSON per story) |
| `--thinking` | Reasoning models: enable `enable_thinking=True` in the chat template and strip `<think>…</think>` from saved outputs |
| `--tensor-parallel-size` | 1 by default; 4 for Llama-4-Scout |
| `--quantization` | Reserved for FP8, unused in current runs |
| `--max-tokens-mult` | Multiplier on word-based max-tokens budget (1.3 default; 2.0 for reasoning+thinking) |
| `--limit-stories` | Smoke-test only: process first N discovered stories |

**Prompt modes the script selects automatically**:

- **Base models** (`BASE_MODELS` set): pure `{story_so_far}` — no system prompt, no chat template. Pure next-token prediction.
- **Instruct / reasoning**: system prompt + user prompt via `tokenizer.apply_chat_template`.
- **Reasoning + `--thinking`**: `enable_thinking=True`, `<think>…</think>` stripped from saved output. `max_tokens = int(words × 1.3 × mult) + 2048` (additive thinking budget) with upper clamp 8192 — required so short-target positions (90% of short stories) don't truncate mid-thought.

Model class membership is declared by the `BASE_MODELS`, `REASONING_MODELS`, `VLM_MODELS`, and `MISTRAL_MODELS` sets near the top of the script. VLM models receive `limit_mm_per_prompt={"image": 0}`. Mistral models receive `tokenizer_mode=config_format=load_format="mistral"`.

### sbatch files — 20 production jobs

| # | File | Model | GPUs | Notes |
|---|---|---|---|---|
| 1 | [run_qwen25_32b_base.sbatch](continuation_dynamic/run_qwen25_32b_base.sbatch) | `Qwen/Qwen2.5-32B` | 1 | base |
| 2 | [run_qwen25_32b_instruct.sbatch](continuation_dynamic/run_qwen25_32b_instruct.sbatch) | `Qwen/Qwen2.5-32B-Instruct` | 1 | |
| 3 | [run_mistral_small_24b_base.sbatch](continuation_dynamic/run_mistral_small_24b_base.sbatch) | `mistralai/Mistral-Small-24B-Base-2501` | 1 | base |
| 4 | [run_mistral_small_24b_instruct.sbatch](continuation_dynamic/run_mistral_small_24b_instruct.sbatch) | `mistralai/Mistral-Small-24B-Instruct-2501` | 1 | |
| 5 | [run_gemma4_31b_base.sbatch](continuation_dynamic/run_gemma4_31b_base.sbatch) | `google/gemma-4-31B` | 1 | base, VLM |
| 6 | [run_gemma4_31b_it.sbatch](continuation_dynamic/run_gemma4_31b_it.sbatch) | `google/gemma-4-31B-it` | 1 | VLM |
| 7 | [run_llama4_scout_base.sbatch](continuation_dynamic/run_llama4_scout_base.sbatch) | `meta-llama/Llama-4-Scout-17B-16E` | 4 | base, VLM, TP=4, gated |
| 8 | [run_llama4_scout_instruct.sbatch](continuation_dynamic/run_llama4_scout_instruct.sbatch) | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 4 | VLM, TP=4, gated |
| 9 | [run_qwen3_32b_nothinking.sbatch](continuation_dynamic/run_qwen3_32b_nothinking.sbatch) | `Qwen/Qwen3-32B` | 1 | reasoning, no thinking |
| 10 | [run_qwen3_32b_thinking.sbatch](continuation_dynamic/run_qwen3_32b_thinking.sbatch) | `Qwen/Qwen3-32B` | 1 | reasoning, `--thinking`, mult=2.0 |
| 11 | [run_qwen35_35b_a3b_nothinking.sbatch](continuation_dynamic/run_qwen35_35b_a3b_nothinking.sbatch) | `Qwen/Qwen3.5-35B-A3B` | 1 | reasoning, MoE |
| 12 | [run_qwen35_35b_a3b_thinking.sbatch](continuation_dynamic/run_qwen35_35b_a3b_thinking.sbatch) | `Qwen/Qwen3.5-35B-A3B` | 1 | reasoning, MoE, `--thinking`, mult=2.0 |
| 13 | [run_qwq_32b.sbatch](continuation_dynamic/run_qwq_32b.sbatch) | `Qwen/QwQ-32B` | 1 | reasoning-only, always thinks (no toggle), mult=2.0 |
| 14 | [run_olmo3_base.sbatch](continuation_dynamic/run_olmo3_base.sbatch) | `allenai/Olmo-3-1125-32B` | 1 | base |
| 15 | [run_olmo31_instruct_sft.sbatch](continuation_dynamic/run_olmo31_instruct_sft.sbatch) | `allenai/Olmo-3.1-32B-Instruct-SFT` | 1 | instruct, SFT stage |
| 16 | [run_olmo31_instruct_dpo.sbatch](continuation_dynamic/run_olmo31_instruct_dpo.sbatch) | `allenai/Olmo-3.1-32B-Instruct-DPO` | 1 | instruct, DPO stage |
| 17 | [run_olmo31_instruct.sbatch](continuation_dynamic/run_olmo31_instruct.sbatch) | `allenai/Olmo-3.1-32B-Instruct` | 1 | instruct, final (RLVR) |
| 18 | [run_olmo3_think_sft.sbatch](continuation_dynamic/run_olmo3_think_sft.sbatch) | `allenai/Olmo-3-32B-Think-SFT` | 1 | reasoning, SFT, always thinks, mult=2.0 |
| 19 | [run_olmo3_think_dpo.sbatch](continuation_dynamic/run_olmo3_think_dpo.sbatch) | `allenai/Olmo-3-32B-Think-DPO` | 1 | reasoning, DPO, always thinks, mult=2.0 |
| 20 | [run_olmo31_think.sbatch](continuation_dynamic/run_olmo31_think.sbatch) | `allenai/Olmo-3.1-32B-Think` | 1 | reasoning, final (RL), always thinks, mult=2.0 |

### Comparison groups

- **Base vs Instruct** (same architecture, different post-training): runs 1↔2, 3↔4, 5↔6, 7↔8
- **Instruct vs Reasoning** (same model family, thinking off vs on): runs 9↔10, 11↔12
- **Reasoning-only** (always-thinking, no instruct analogue): run 13 (QwQ-32B)
- **Olmo-3 post-training staircase** (same base, progressive stages): run 14 (base) → 15 (Instruct-SFT) → 16 (Instruct-DPO) → 17 (Instruct-RLVR); and in parallel 14 (base) → 18 (Think-SFT) → 19 (Think-DPO) → 20 (Think-RL). Lets you isolate the effect of each post-training phase on narrative-continuation behavior within a single model family. Olmo-Think variants are always-thinking (chat template pre-seeds `<think>`), so `strip_thinking()` runs on every output.

### Submit

```bash
cd continuation_dynamic
bash submit_all.sh               # submits all 12
# or individually:
sbatch run_qwen25_32b_base.sbatch
```

### sbatch environment boilerplate

Every sbatch loads `python/miniforge3_pytorch` and activates `/projects/bgye/envs/llm` (vLLM 0.19.1, PyTorch 2.10, transformers 5.6). Key env exports:

- `HF_HOME=/projects/bgye/models/hf_cache` — shared HuggingFace cache
- `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` — fast NVMe scratch on `/work/nvme/bgye/yzhu38/cache/`
- `VLLM_CACHE_ROOT=/work/nvme/bgye/yzhu38/cache/vllm-$SLURM_JOB_ID` — **per-job** torch-compile cache (prevents cross-job races when the shared home is blown away)
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` — force FlashAttention, bypass flashinfer (needed for Qwen3.5-35B-A3B and Llama-4-Scout where flashinfer's GDN CUTLASS kernels fail to compile on this cluster's nvcc)

### Output schema

Each `results_<short_name>/<sid>_continuations.json` contains:

```jsonc
{
  "story_id": "00002",
  "source_file": "...",
  "n_sentences": 134,
  "n_words": 2267,
  "n_positions_evaluated": 4,
  "position_pcts_config": [0.4, 0.6, 0.8, 0.9],
  "n_continuations_per_position": 5,
  "temperature": 1.2,
  "top_p": 0.95,
  "model": "Qwen/Qwen3-32B",
  "prompt_mode": "chat",          // "chat" or "base_raw"
  "thinking_enabled": true,
  "tensor_parallel_size": 1,
  "quantization": null,
  "max_tokens_mult": 2.0,
  "positions": [
    {
      "position": 54,
      "pct_story_revealed": 40.3,
      "target_n_words": 1315,
      "prefix": "<first 40% of the story>",
      "ground_truth_continuation": "<remaining 60%>",
      "continuations": ["<gen 1>", "<gen 2>", "<gen 3>", "<gen 4>", "<gen 5>"]
    },
    // ... positions 60%, 80%, 90%
  ]
}
```

### Smoke tests

[smoke_qwen25_32b_base.sbatch](continuation_dynamic/smoke_qwen25_32b_base.sbatch) and [smoke_qwen3_32b_thinking.sbatch](continuation_dynamic/smoke_qwen3_32b_thinking.sbatch) run `--limit-stories 2` against representative models (base-prompt path and thinking path). Results go to [smoke_results/](continuation_dynamic/smoke_results/). Useful for validating env changes without committing a 48-hour slot.

### Resume semantics

`run_continuations.py` skips any story whose output JSON already exists in the target `results_*/` directory, so interrupted jobs can be resubmitted safely without redoing work.

### Monitoring

```bash
squeue -u $USER                                     # queue state
sacct -j <JOB_ID> -X -o 'JobID,State,Elapsed,ExitCode'
tail -f continuation_dynamic/logs/cont_<short>_<jobid>.out   # live output
ls continuation_dynamic/results_<short>/ | wc -l    # progress (JSONs written)
```

---

## Data: `NEWCORPUS_CLEANED/`

5001 `.txt` files (story_ids `00001.txt` … `05001.txt`), one story per file. Used directly as `--data-dir` input. Stories over 5000 words are skipped by the generator's `MAX_WORD_COUNT` filter; stories with < 5 sentences are skipped at the position-selection stage.

The `MAX_STORY_ID` filter (default 200) is bypassed when sbatch passes `--no-id-filter`, which is the production setting. That makes all numeric-ID stories eligible.

---

## Cluster environment (NCSA DeltaAI)

- **Partition**: `ghx4`, 150 nodes, 4× NVIDIA GH200 Grace-Hopper Superchip per node. **GPU-side HBM is 96 GiB HBM3 per GPU**; Slurm's `nvidia_gh200_120gb` label refers to the Grace CPU's LPDDR5X pool (unified memory, not HBM). aarch64 Grace CPU, 288 cores, 870 GB RAM per node (gh001–002 have 488 GB).
- **Account**: `bgye-dtai-gh`
- **Conda env**: `/projects/bgye/envs/llm`
- **Filesystems**: home (100 GB, scrubbed every 90 days) for configs only; `/projects/bgye/` shared team workspace (where this tree lives); `/work/nvme/bgye/yzhu38/cache/` for fast-scratch compile caches
- **Budget**: 3000 GPU-hours total for the team — full 12-run slate is ~300–500 GPU-hrs estimated
- **Cluster docs**: https://my-garden-eight-kappa.vercel.app/apto-general/apto-compute/delta-ai-onboarding-guide/

---

## Changelog

- **2026-04** — Project rewrite. Replaced v1–v4 close-reading + distance pipeline with the current 12-model continuation study. All prior scripts, notebooks, and intermediate data archived under `archive/2026-04-cleanup/`.
