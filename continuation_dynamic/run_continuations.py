#!/usr/bin/env python3
"""
Dynamic-length story continuation generation for NEWCORPUS stories.

Extension of run_continuation_old.py. Same pipeline (sentence split,
fixed-percent positions, dynamic max_tokens, resume-safe per-story
JSON), with:
  - configurable model via --model
  - 4 positions (40/60/80/90%) instead of 5 (dropped 20%)
  - 5 continuations per position instead of 20
  - base-model prompt path: raw prefix only (no instruction, no chat)
  - --thinking flag for reasoning models (enable_thinking + <think> strip)
  - VLM / Mistral / Llama-4 Scout specific kwargs
  - --tensor-parallel-size, --quantization, --max-tokens-mult

Usage:
    python run_continuations.py --model Qwen/Qwen2.5-32B \
        --results-dir ./results_qwen25_32b_base --no-id-filter
"""

import os, re, json, time, gc, argparse

# Patch for vLLM + transformers 5.x compatibility (inherited from old script)
from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "/projects/bgye/yzhu38/narrative_project/NEWCORPUS_CLEANED"
RESULTS_DIR = "/projects/bgye/yzhu38/narrative_project/continuation_dynamic/results"

N_CONTINUATIONS = 5
TEMPERATURE = 1.2
TOP_P = 0.95

MAX_STORY_ID = 200
MAX_WORD_COUNT = 5000

MAX_MODEL_LEN = 32768

POSITION_PCTS = [0.40, 0.60, 0.80, 0.90]

# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------

BASE_MODELS = {
    "Qwen/Qwen2.5-32B",
    "mistralai/Mistral-Small-24B-Base-2501",
    "google/gemma-4-31B",
    "meta-llama/Llama-4-Scout-17B-16E",
}

REASONING_MODELS = {
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3.5-35B-A3B",
}

VLM_MODELS = {
    "google/gemma-4-31B",
    "google/gemma-4-31B-it",
    "meta-llama/Llama-4-Scout-17B-16E",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Qwen/Qwen3.5-35B-A3B",
}

MISTRAL_MODELS = {
    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mistral-Small-24B-Instruct-2501",
}


def is_base_model(mid):       return mid in BASE_MODELS
def is_reasoning_model(mid):  return mid in REASONING_MODELS
def is_vlm_model(mid):        return mid in VLM_MODELS
def is_mistral_model(mid):    return mid in MISTRAL_MODELS


# ---------------------------------------------------------------------------
# Prompts (same wording as run_continuation_old.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a fiction writer. Continue the story naturally in the "
    "same style and voice. Write only story text — no commentary, "
    "no meta-discussion, no preamble, no quotation marks around "
    "your continuation."
)

COMPLETION_TEMPLATE = """\
Continue this story to its conclusion in approximately {n_words} words. \
Maintain the same tone, style, and narrative voice throughout. \
Do not summarize or describe what happens — write the actual story \
text as it would appear on the page.

STORY SO FAR:
{story_so_far}"""

# Base models: raw story prefix only. No instruction.
BASE_PROMPT_TEMPLATE = "{story_so_far}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text):
    return _THINK_RE.sub("", text).lstrip()


def discover_stories(data_dir, max_id=MAX_STORY_ID, max_words=MAX_WORD_COUNT,
                     no_id_filter=False, min_id=0):
    stories = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        story_id = fname.replace(".txt", "")
        if not no_id_filter:
            try:
                id_num = int(story_id)
            except ValueError:
                continue
            if id_num < min_id or id_num >= max_id:
                continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text.split()) > max_words:
            continue
        stories[story_id] = fpath
    return stories


def split_sentences(text):
    boundaries = [0]
    for m in re.finditer(
        r'[.!?]["”’)]*\s+(?=[A-Z“"(\[])', text
    ):
        boundaries.append(m.end())
    parts = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def select_positions(n_sentences, pcts=POSITION_PCTS):
    positions = set()
    for pct in pcts:
        pos = max(1, min(int(round(pct * n_sentences)), n_sentences - 1))
        positions.add(pos)
    return sorted(positions)


def get_llm_kwargs(model_id, tensor_parallel_size=1, quantization=None,
                   gpu_util=0.92):
    kwargs = dict(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
    )
    if quantization:
        kwargs["quantization"] = quantization
    if is_vlm_model(model_id):
        kwargs["limit_mm_per_prompt"] = {"image": 0}
    if is_mistral_model(model_id):
        kwargs["tokenizer_mode"] = "mistral"
        kwargs["config_format"] = "mistral"
        kwargs["load_format"] = "mistral"
    return kwargs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HF model id, e.g. Qwen/Qwen2.5-32B")
    parser.add_argument("--stories", type=str, default=None,
                        help="Comma-separated story IDs (e.g. 00015,00056)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--no-id-filter", action="store_true")
    parser.add_argument("--min-id", type=int, default=0)
    parser.add_argument("--max-id", type=int, default=MAX_STORY_ID)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--quantization", type=str, default=None,
                        help="e.g. fp8 (None = full bf16)")
    parser.add_argument("--thinking", action="store_true",
                        help="Reasoning-model switch. Enables thinking in "
                             "chat template and strips <think>...</think>.")
    parser.add_argument("--max-tokens-mult", type=float, default=1.3,
                        help="Extra multiplier on dynamic max_tokens. "
                             "Use 2.0 for reasoning models with thinking on "
                             "to leave room for the hidden thought stream.")
    parser.add_argument("--gpu-util", type=float, default=0.92)
    parser.add_argument("--limit-stories", type=int, default=0,
                        help="If > 0, only process the first N discovered "
                             "stories. For smoke-tests.")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_id = args.model
    data_dir = args.data_dir
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    use_base_prompt = is_base_model(model_id)
    is_reasoning = is_reasoning_model(model_id)
    enable_thinking = is_reasoning and args.thinking
    strip_think = is_reasoning and args.thinking
    prompt_mode = "base_raw" if use_base_prompt else "chat"

    print("=== run_continuations.py ===")
    print(f"  model                = {model_id}")
    print(f"  prompt_mode          = {prompt_mode}")
    print(f"  is_base              = {use_base_prompt}")
    print(f"  is_reasoning         = {is_reasoning}")
    print(f"  thinking (flag)      = {args.thinking}")
    print(f"  enable_thinking      = {enable_thinking}")
    print(f"  strip_think          = {strip_think}")
    print(f"  tensor_parallel_size = {args.tensor_parallel_size}")
    print(f"  quantization         = {args.quantization}")
    print(f"  max_tokens_mult      = {args.max_tokens_mult}")
    print(f"  n_continuations      = {N_CONTINUATIONS}")
    print(f"  positions            = {POSITION_PCTS}")
    print(f"  results_dir          = {results_dir}")

    stories = discover_stories(
        data_dir,
        max_id=args.max_id,
        no_id_filter=args.no_id_filter,
        min_id=args.min_id,
    )
    print(f"Discovered {len(stories)} stories in {data_dir} "
          f"(<= {MAX_WORD_COUNT} words)")

    if args.stories:
        ids = [s.strip().zfill(5) for s in args.stories.split(",")]
        stories = {k: v for k, v in stories.items() if k in ids}
        print(f"Filtered to {len(stories)} stories: {list(stories.keys())}")

    if args.n_shards > 1:
        all_keys = sorted(stories.keys())
        chunk_size = len(all_keys) // args.n_shards
        remainder = len(all_keys) % args.n_shards
        start = sum(chunk_size + (1 if i < remainder else 0)
                    for i in range(args.shard_id))
        my_size = chunk_size + (1 if args.shard_id < remainder else 0)
        my_keys = all_keys[start:start + my_size]
        stories = {k: stories[k] for k in my_keys}
        print(f"Shard {args.shard_id}/{args.n_shards}: {len(stories)} stories "
              f"(IDs {my_keys[0]}–{my_keys[-1]})")

    if args.limit_stories > 0:
        keys = sorted(stories.keys())[:args.limit_stories]
        stories = {k: stories[k] for k in keys}
        print(f"Smoke-test: limited to first {len(stories)} stories: {keys}")

    llm_kwargs = get_llm_kwargs(
        model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        gpu_util=args.gpu_util,
    )
    print(f"Loading model with kwargs: {llm_kwargs}")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    processed = 0
    skipped = 0
    t_start = time.time()

    for story_id, story_path in stories.items():
        output_path = os.path.join(results_dir, f"{story_id}_continuations.json")

        if os.path.exists(output_path):
            skipped += 1
            continue

        with open(story_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()

        sentences = split_sentences(full_text)
        n_sents = len(sentences)

        if n_sents < 5:
            continue

        positions = select_positions(n_sents)

        print(f"\n{'=' * 70}")
        print(f"[{story_id}]")
        print(f"  {n_sents} sentences, {len(full_text.split())} words")
        print(f"  Positions: {positions} "
              f"(={[round(p / n_sents * 100) for p in positions]}%)")

        position_results = []

        for pos in positions:
            prefix = " ".join(sentences[:pos])
            ground_truth = " ".join(sentences[pos:])
            target_n_words = len(ground_truth.split())

            # Dynamic max_tokens: words * 1.3 (word->token) * mult (buffer).
            # When thinking is on we also add a fixed thinking budget so that
            # short-target positions (e.g. 90% of short stories) still have
            # room for the reasoning phase before running out of tokens.
            thinking_budget = 2048 if strip_think else 0
            upper_clamp = 8192 if strip_think else 4096
            dynamic_max_tokens = int(target_n_words * 1.3 * args.max_tokens_mult) \
                + thinking_budget
            dynamic_max_tokens = max(dynamic_max_tokens, 100 + thinking_budget)
            dynamic_max_tokens = min(dynamic_max_tokens, upper_clamp)

            if use_base_prompt:
                prompt = BASE_PROMPT_TEMPLATE.format(story_so_far=prefix)
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": COMPLETION_TEMPLATE.format(
                        story_so_far=prefix,
                        n_words=target_n_words,
                    )},
                ]
                chat_kwargs = dict(
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if is_reasoning:
                    chat_kwargs["enable_thinking"] = enable_thinking
                prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)

            prompt_tokens = len(tokenizer.encode(prompt))
            if prompt_tokens + dynamic_max_tokens > MAX_MODEL_LEN:
                print(f"  pos={pos} SKIPPED: {prompt_tokens} prompt tokens "
                      f"+ {dynamic_max_tokens} gen tokens > {MAX_MODEL_LEN}")
                continue

            sp = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=dynamic_max_tokens,
                n=N_CONTINUATIONS,
            )

            t0 = time.time()
            outputs = llm.generate([prompt], sp)
            elapsed_pos = time.time() - t0

            raw = [o.text for o in outputs[0].outputs]
            if strip_think:
                continuations = [strip_thinking(t).strip() for t in raw]
            else:
                continuations = [t.strip() for t in raw]

            pct = round(pos / n_sents * 100, 1)
            print(f"  pos={pos} ({pct}%): "
                  f"target={target_n_words}w, "
                  f"max_tok={dynamic_max_tokens}, "
                  f"{elapsed_pos:.1f}s")

            position_results.append({
                "position": pos,
                "n_context_sentences": pos,
                "n_remaining_sentences": n_sents - pos,
                "pct_story_revealed": pct,
                "target_n_words": target_n_words,
                "prefix": prefix,
                "ground_truth_continuation": ground_truth,
                "continuations": continuations,
            })

        result = {
            "story_id": story_id,
            "source_file": story_path,
            "n_sentences": n_sents,
            "n_words": len(full_text.split()),
            "n_positions_evaluated": len(positions),
            "position_pcts_config": POSITION_PCTS,
            "n_continuations_per_position": N_CONTINUATIONS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "model": model_id,
            "prompt_mode": prompt_mode,
            "thinking_enabled": enable_thinking,
            "tensor_parallel_size": args.tensor_parallel_size,
            "quantization": args.quantization,
            "max_tokens_mult": args.max_tokens_mult,
            "positions": position_results,
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        processed += 1
        elapsed = time.time() - t_start
        rate = processed / elapsed * 60 if elapsed > 0 else 0
        print(f"  [{story_id}] Saved. Total: {processed} done, "
              f"{skipped} skipped, {rate:.1f} stories/min")

        gc.collect()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE. Processed: {processed}, Skipped: {skipped}, "
          f"Time: {elapsed / 3600:.1f}h")


if __name__ == "__main__":
    main()
