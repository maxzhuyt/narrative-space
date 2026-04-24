#!/usr/bin/env python3
"""
Validate vLLM model loading and inference for all models in the comparison.

Tests:
  1. Model classification consistency (base / instruct / reasoning)
  2. Tokenizer loading and chat template presence
  3. Chat template structure (enable_thinking=False on reasoning models, etc.)
  4. Thinking mode toggle for Qwen3 / Qwen3.5
  5. strip_thinking helper behavior
  6. Short generation test on Mistral-Small-24B-Instruct

Run on a GPU compute node. Easiest is via sbatch — the project's sbatch
wrappers already set HF_TOKEN via `source .env` and activate the conda env.
Manual invocation:

  module load python/miniforge3_pytorch
  conda activate /projects/bgye/envs/llm
  set -a; source /projects/bgye/yzhu38/narrative_project/.env; set +a
  python validate_models.py
"""

import os
import sys

os.environ.setdefault("HF_HOME", "/projects/bgye/models/hf_cache")

# HF_TOKEN must be provided by the caller's shell environment — typically
# via `set -a; source /projects/bgye/yzhu38/narrative_project/.env; set +a`
# in the sbatch wrapper. huggingface_hub reads it from os.environ automatically.
if not os.environ.get("HF_TOKEN"):
    sys.exit("HF_TOKEN not set. Run `source .env` (project root) before this script.")

# Models covered by the current 12-run comparison. If you add a model to
# run_continuations.py's classification sets, mirror it here.
MODELS = [
    ("Qwen/Qwen2.5-32B",                           "base",      False),
    ("Qwen/Qwen2.5-32B-Instruct",                  "instruct",  False),
    ("mistralai/Mistral-Small-24B-Base-2501",      "base",      False),
    ("mistralai/Mistral-Small-24B-Instruct-2501",  "instruct",  False),
    ("google/gemma-4-31B",                         "base",      False),
    ("google/gemma-4-31B-it",                      "instruct",  False),
    ("meta-llama/Llama-4-Scout-17B-16E",           "base",      False),
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct",  "instruct",  False),
    ("Qwen/Qwen3-32B",                             "reasoning", False),
    ("Qwen/Qwen3.5-35B-A3B",                       "reasoning", False),
    ("Qwen/QwQ-32B",                               "reasoning", False),
]

sys.path.insert(0, "/projects/bgye/yzhu38/narrative_project/continuation_dynamic")
from run_continuations import is_base_model, is_reasoning_model, strip_thinking

import re

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"

PASSED = 0
FAILED = 0

def test(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        print(f"  [PASS] {name}")
        PASSED += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        FAILED += 1

print("=" * 70)
print("MODEL VALIDATION TEST")
print("=" * 70)

# Test 1: Model classification
print("\n--- Test 1: Model Classification ---")
for model_id, expected_type, _ in MODELS:
    is_base = is_base_model(model_id)
    is_reason = is_reasoning_model(model_id)
    if expected_type == "base":
        test(f"{model_id} is base", is_base and not is_reason,
             f"is_base={is_base}, is_reasoning={is_reason}")
    elif expected_type == "instruct":
        test(f"{model_id} is instruct", not is_base and not is_reason,
             f"is_base={is_base}, is_reasoning={is_reason}")
    elif expected_type == "reasoning":
        test(f"{model_id} is reasoning", is_reason and not is_base,
             f"is_base={is_base}, is_reasoning={is_reason}")

# Test 2: Tokenizer loading
print("\n--- Test 2: Tokenizer Loading ---")
from transformers import AutoTokenizer

tokenizer_results = {}
for model_id, model_type, _ in MODELS:
    try:
        kwargs = dict(trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
        tokenizer_results[model_id] = tok
        test(f"{model_id} tokenizer loaded", True)

        has_chat = hasattr(tok, "chat_template") and tok.chat_template is not None
        if model_type == "base":
            test(f"{model_id} base model chat_template status",
                 True,  # just note what it is
                 f"has_chat_template={has_chat}")
        else:
            test(f"{model_id} has chat_template", has_chat,
                 f"has_chat_template={has_chat}")
    except Exception as e:
        test(f"{model_id} tokenizer loaded", False, str(e))

# Test 3: Chat template structure
print("\n--- Test 3: Chat Template Structure ---")
msgs = [
    {"role": "system", "content": "You are a fiction writer."},
    {"role": "user", "content": "Continue this story."},
]

for model_id, model_type, _ in MODELS:
    if model_id not in tokenizer_results:
        continue
    tok = tokenizer_results[model_id]

    if model_type == "base":
        print(f"  [SKIP] {model_id} is base model, no chat template test")
        continue

    try:
        if model_type == "reasoning":
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            has_think = THINK_OPEN in prompt or THINK_CLOSE in prompt
            test(f"{model_id} enable_thinking=False no think tags",
                 not has_think, f"found think tags in prompt")
        else:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            test(f"{model_id} chat template works", len(prompt) > 0,
                 f"prompt length={len(prompt)}")

        print(f"    Preview: {prompt[:120]}...")
    except Exception as e:
        test(f"{model_id} chat template", False, str(e))

# Test 4: Thinking mode toggle for Qwen3 models
print("\n--- Test 4: Thinking Mode Toggle ---")
for qwen_model in ["Qwen/Qwen3-32B", "Qwen/Qwen3.5-35B-A3B"]:
    if qwen_model not in tokenizer_results:
        print(f"  [SKIP] {qwen_model} not loaded")
        continue
    tok = tokenizer_results[qwen_model]
    try:
        p_think = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
        p_no_think = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        has_think_on = THINK_OPEN in p_think or THINK_CLOSE in p_think
        has_think_off = THINK_OPEN in p_no_think or THINK_CLOSE in p_no_think
        test(f"{qwen_model} enable_thinking=True has think tags", has_think_on,
             f"no think tags found in thinking prompt")
        test(f"{qwen_model} enable_thinking=False no think tags",
             not has_think_off, f"found think tags when disabled")
    except Exception as e:
        test(f"{qwen_model} thinking toggle", False, str(e))

# Test 5: strip_thinking function
print("\n--- Test 5: strip_thinking Function ---")
test_text = f"Here is my reasoning: {THINK_OPEN}I think the story should continue with a twist{THINK_CLOSE} The twist was unexpected."
stripped = strip_thinking(test_text)
test("strip_thinking removes think blocks",
     THINK_OPEN not in stripped and THINK_CLOSE not in stripped,
     f"stripped={repr(stripped[:80])}")
test("strip_thinking preserves text outside blocks",
     "The twist was unexpected." in stripped,
     f"stripped={repr(stripped)}")

# Multi-line think block (simulates a reasoning model's raw output)
long_output = f"{THINK_OPEN}\nLet me think about this story.\nThe cat sat on the mat.\n{THINK_CLOSE}\nThe dog bounded through the door."
stripped_long = strip_thinking(long_output)
test("multi-line think block stripped correctly",
     THINK_OPEN not in stripped_long and THINK_CLOSE not in stripped_long,
     f"stripped={repr(stripped_long[:80])}")

# Test 6: vLLM generation test
print("\n--- Test 6: Full Generation Test (Mistral-Small-24B-Instruct) ---")
try:
    from vllm import LLM, SamplingParams
    import time

    t0 = time.time()
    llm = LLM(
        model="mistralai/Mistral-Small-24B-Instruct-2501",
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
    )
    load_time = time.time() - t0
    test("Mistral-Small-24B-Instruct model loaded", True, f"in {load_time:.1f}s")

    tokenizer = llm.get_tokenizer()
    messages = [
        {"role": "system", "content": "You are a fiction writer."},
        {"role": "user", "content": "Continue this story in ~50 words.\n\nSTORY SO FAR:\nThe cat sat on the mat."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100, n=2)
    outputs = llm.generate([prompt], sp)

    test("Generation produced 2 outputs", len(outputs[0].outputs) == 2,
         f"got {len(outputs[0].outputs)} outputs")

    for i, out in enumerate(outputs[0].outputs):
        text = out.text.strip()
        print(f"    Output {i+1}: {text[:100]}...")
        test(f"Output {i+1} is non-empty", len(text) > 0, "empty output")

    del llm
    print("    Model freed OK")

except Exception as e:
    test("Mistral-Small-24B-Instruct full generation", False, str(e))

# Summary
print("\n" + "=" * 70)
print(f"RESULTS: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED} tests")
if FAILED > 0:
    print("SOME TESTS FAILED - review output above before submitting jobs")
    sys.exit(1)
else:
    print("ALL TESTS PASSED - ready to submit jobs")
    sys.exit(0)