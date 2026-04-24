#!/usr/bin/env python3
"""
Validate vLLM model loading and inference for all models in the comparison.

Tests:
  1. Tokenizer loading and chat template for each model
  2. enable_thinking=False works for Qwen3/3.5 reasoning models
  3. Short generation test for one model (Mistral-Small-24B-Instruct)
  4. Thinking tag stripping verification
  5. Base model detection (no chat template)
  6. Model classification consistency

Run on a GPU compute node after:
  salloc --partition=ghx4 --account=bgye --gpus-per-node=1 --cpus-per-task=16 --mem=64G --time=01:00:00
  ssh <compute_node>
  module load python/miniforge3_pytorch
  conda activate /projects/bgye/envs/llm
  python validate_models.py
"""

import os
import sys

os.environ["HF_HOME"] = "/projects/bgye/models/hf_cache"
os.environ["HF_TOKEN"] = ""

MODELS = [
    ("Qwen/Qwen2.5-32B",                        "base",      False),
    ("Qwen/Qwen2.5-32B-Instruct",                "instruct",  False),
    ("mistralai/Mistral-Small-24B-Base-2501",     "base",      False),
    ("mistralai/Mistral-Small-24B-Instruct-2501","instruct",  False),
    ("google/gemma-4-31B",                        "base",      False),
    ("google/gemma-4-31B-it",                     "instruct",  False),
    ("Qwen/Qwen3-32B",                           "reasoning", False),
    ("Qwen/Qwen3-30B-A3B",                       "reasoning", False),
    ("Qwen/Qwen3.5-35B-A3B",                     "reasoning", False),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "reasoning", False),
]

sys.path.insert(0, "/projects/bgye/yzhu38/narrative_project/continuation_dynamic")
from run_continuations import model_shortname, is_base_model, is_reasoning_model, strip_thinking

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
    shortname = model_shortname(model_id)
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
        if model_type == "reasoning" and "DeepSeek-R1" not in model_id:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
            has_think = THINK_OPEN in prompt or THINK_CLOSE in prompt
            test(f"{model_id} enable_thinking=False no think tags",
                 not has_think, f"found think tags in prompt")
        elif "DeepSeek-R1" in model_id:
            ds_msgs = [{"role": "user", "content": "Continue this story."}]
            prompt = tok.apply_chat_template(
                ds_msgs, tokenize=False, add_generation_prompt=True
            )
            test(f"{model_id} DeepSeek chat template works",
                 len(prompt) > 0, f"prompt length={len(prompt)}")
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
for qwen_model in ["Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3.5-35B-A3B"]:
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

# Simulate DeepSeek-R1 output
ds_output = f"{THINK_OPEN}\nLet me think about this story.\nThe cat sat on the mat.\n{THINK_CLOSE}\nThe dog bounded through the door."
stripped_ds = strip_thinking(ds_output)
test("DeepSeek-R1 style output stripped correctly",
     THINK_OPEN not in stripped_ds and THINK_CLOSE not in stripped_ds,
     f"stripped={repr(stripped_ds[:80])}")

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