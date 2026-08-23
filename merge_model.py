"""Merge the trained LoRA adapter into the base model.

Runs on the RunPod pod, right after fine_tune.py completes (before you stop
the pod — merging needs the full-size base model in memory). Loads LLaMA
3.1 8B Instruct in fp16, layers the trained adapter from ./adapter on top
with PeftModel, folds the adapter mathematics permanently into the base
weights with merge_and_unload, and saves one standard, deployable model
directory with no PEFT overhead.

Merging happens in fp16, never in 4-bit: merging into already-rounded
QLoRA weights would bake quantisation error into the exact behaviour the
training paid for. For every adapted layer:
    W_merged = W_original + B x A x (alpha / r)
then the adapter matrices are discarded entirely.

Usage:
    python merge_model.py
Then download ./merged-model back to your local machine (or continue
running local_inference.py / evaluate_models.py on the pod itself).
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ADAPTER_PATH = "./adapter"
MERGED_PATH = "./merged-model"  # matches .gitignore and README's documented path

# STEP 1: Load the base model in fp16, NOT 4-bit.
# Merging into quantised weights bakes rounding error into the result;
# a one-time fp16 merge keeps full fidelity.
print("Loading base model for merging (downloads ~16GB on first run)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# STEP 2: Layer the LoRA adapter on top of the base model
print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# STEP 3: Merge and unload.
# "Merge" applies the W + B x A x (alpha/r) formula from the module
# docstring, layer by layer, baking the adapter's learned deltas directly
# into the base weights. "Unload" then drops the PeftModel wrapper
# entirely, since the adapter's math is now redundant -- what's left is
# a plain AutoModelForCausalLM, loadable with no `peft` dependency at all
# (which is exactly what local_inference.py and evaluate_models.py do).
print("Merging adapter weights into base model...")
model = model.merge_and_unload()

# STEP 4: Save the merged model plus its tokeniser
print(f"Saving merged model to {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_PATH)

print("Merge complete.")
print(f"Merged model saved to: {MERGED_PATH}")
print("You can delete the adapter/ folder now if disk space is tight.")
