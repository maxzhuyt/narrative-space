#!/usr/bin/env python3
"""
Dynamic-length story continuation generation for NEWCORPUS stories.

At 5 fixed positions (20%, 40%, 60%, 80%, 90% of story), generates 20
continuations whose target length matches the remaining story text.

Each output entry is self-contained:
  - prefix (the story text used as context)
  - ground_truth_continuation (the actual remainder)
  - continuations[] (the 20 generated alternatives)

This makes the output directly usable as a finetuning dataset.

Based on run_endings.py (corpus discovery, sentence splitting) with
the continuation strategy from run_cwb_continuations.py (dynamic length,
5 positions, 20 continuations, prose-style prompt).

Usage:
    python run_continuations.py
    python run_continuations.py --stories 00015,00056
    python run_continuations.py --data-dir /path/to/stories --min-id 0 --max-id 1000
"""

import os, sys, re, json, time, gc, argparse
import numpy as np

# Patch for vLLM + transformers 5.x compatibility
from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "/project/jevans/maxzhuyt/models/Qwen3-32B"
DATA_DIR = "/project/jevans/maxzhuyt/narrative_project/NEWCORPUS_CLEANED"
RESULTS_DIR = "/project/jevans/maxzhuyt/narrative_project/continuation_dynamic"

N_CONTINUATIONS = 20
TEMPERATURE = 1.2
TOP_P = 0.95

MAX_STORY_ID = 200
MAX_WORD_COUNT = 5000
MIN_SENT_WORDS = 10

# 5 fixed positions: what % of the story is revealed as context
POSITION_PCTS = [0.20, 0.40, 0.60, 0.80, 0.90]

# ---------------------------------------------------------------------------
# Prompts
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        r'[.!?]["\u201d\u2019)]*\s+(?=[A-Z\u201c"(\[])', text
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
    """Convert percentage positions to sentence indices (1-indexed, deduplicated)."""
    positions = set()
    for pct in pcts:
        pos = max(1, min(int(round(pct * n_sentences)), n_sentences - 1))
        positions.add(pos)
    return sorted(positions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=str, default=None,
                        help="Comma-separated story IDs (e.g. 00015,00056)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Positions to batch per vLLM call")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--no-id-filter", action="store_true")
    parser.add_argument("--min-id", type=int, default=0)
    parser.add_argument("--max-id", type=int, default=MAX_STORY_ID)
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index (0-based)")
    parser.add_argument("--n-shards", type=int, default=1,
                        help="Total number of shards")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    data_dir = args.data_dir
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    stories = discover_stories(data_dir, max_id=args.max_id,
                               no_id_filter=args.no_id_filter,
                               min_id=args.min_id)
    print(f"Discovered {len(stories)} stories in {data_dir} "
          f"(<= {MAX_WORD_COUNT} words)")

    if args.stories:
        ids = [s.strip().zfill(5) for s in args.stories.split(",")]
        stories = {k: v for k, v in stories.items() if k in ids}
        print(f"Filtered to {len(stories)} stories: {list(stories.keys())}")

    # ── Shard (contiguous chunks) ──────────────────────────────────────
    if args.n_shards > 1:
        all_keys = sorted(stories.keys())
        chunk_size = len(all_keys) // args.n_shards
        remainder = len(all_keys) % args.n_shards
        # First 'remainder' shards get chunk_size+1, rest get chunk_size
        start = sum(chunk_size + (1 if i < remainder else 0)
                    for i in range(args.shard_id))
        my_size = chunk_size + (1 if args.shard_id < remainder else 0)
        my_keys = all_keys[start:start + my_size]
        stories = {k: stories[k] for k in my_keys}
        print(f"Shard {args.shard_id}/{args.n_shards}: {len(stories)} stories "
              f"(IDs {my_keys[0]}–{my_keys[-1]})")

    # ── Load model ─────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=32768,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    processed = 0
    skipped = 0
    t_start = time.time()

    for story_id, story_path in stories.items():
        output_path = os.path.join(results_dir, f"{story_id}_continuations.json")

        # Resume-safe: skip if output already exists
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

        print(f"\n{'='*70}")
        print(f"[{story_id}]")
        print(f"  {n_sents} sentences, {len(full_text.split())} words")
        print(f"  Positions: {positions} "
              f"(={[round(p/n_sents*100) for p in positions]}%)")

        # ── Generate per position ──────────────────────────────────────
        position_results = []

        for pos in positions:
            prefix = " ".join(sentences[:pos])
            ground_truth = " ".join(sentences[pos:])
            target_n_words = len(ground_truth.split())

            # Dynamic max_tokens: 1 word ≈ 1.3 tokens, +30% buffer
            dynamic_max_tokens = int(target_n_words * 1.3 * 1.3)
            dynamic_max_tokens = max(dynamic_max_tokens, 100)
            dynamic_max_tokens = min(dynamic_max_tokens, 4096)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": COMPLETION_TEMPLATE.format(
                    story_so_far=prefix,
                    n_words=target_n_words
                )},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            # Skip if prompt + generation would exceed model context
            prompt_tokens = len(tokenizer.encode(prompt))
            if prompt_tokens + dynamic_max_tokens > 32768:
                print(f"  pos={pos} SKIPPED: {prompt_tokens} prompt tokens "
                      f"+ {dynamic_max_tokens} gen tokens > 32768")
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

            continuations = [o.text.strip() for o in outputs[0].outputs]

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

        # ── Save ───────────────────────────────────────────────────────
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
            "model": MODEL_PATH,
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
    print(f"\n{'='*70}")
    print(f"DONE. Processed: {processed}, Skipped: {skipped}, "
          f"Time: {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
