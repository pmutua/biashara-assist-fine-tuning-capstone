"""QLoRA fine-tuning of LLaMA 3.1 8B Instruct on the BiasharaAssist dataset.

Runs ON A RUNPOD GPU POD (24GB+ VRAM, Ampere-or-newer so bf16 works
unmodified — RTX 4090 / A5000 / A100 / A40), not on a local laptop. Loads
the base model in 4-bit (QLoRA), injects LoRA adapters over the attention
projections, and trains on data/train.jsonl and data/val.jsonl, produced by
this repo's data_prep.py (159 train / 20 val records as of M1).

Pinned library versions (see requirements.txt and the RunPod setup notes
this repo's sibling course carried over): transformers==4.43.3,
trl==0.8.6, peft==0.11.1, bitsandbytes>=0.46.1, accelerate==0.33.0. These
exact pins matter — see the comment above the imports below for why.

Needs a CUDA GPU, bitsandbytes, an approved Meta LLaMA access grant, and a
live HF_TOKEN. Do not run this against a CPU-only environment or without
those three prerequisites confirmed.

Usage (on the pod, inside tmux so the run survives a dropped connection):
    export HF_TOKEN=hf_your_token_here
    tmux new -s finetune
    python fine_tune.py
"""

from __future__ import annotations

import os

# transformers is pinned to 4.43.3: old enough that trl==0.8.6's SFTTrainer
# still accepts its tokenizer= argument, new enough to understand Llama
# 3.1's rope_scaling config format (introduced after transformers==4.41.2).
# bitsandbytes is pinned >=0.46.1 for a CUDA-12.8-compatible binary — not a
# Python-API constraint like the other pins, a hardware/driver one.
# accelerate is pinned to 0.33.0 (same release window as transformers
# 4.43.3) because a newer accelerate's dispatch_model() calls .to() on an
# already-placed 4-bit model, which transformers explicitly forbids.
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# ----------------------------- CONFIGURATION -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # export HF_TOKEN=hf_... before running
BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
OUTPUT_DIR = "adapter"  # matches .gitignore and plot_loss.py's default path
DATA_DIR = "./data"

# --- LoRA configuration ---
LORA_R = 16  # Rank of the adapter matrices — 16 is the standard starting
             # point for an 8B model: enough capacity to learn a narrow
             # domain (4 topic areas) without the memory cost of rank 32+.
LORA_ALPHA = 32  # Scaling factor, conventionally 2x the rank, so the
                  # adapter's effective learning rate stays well-behaved
                  # as r changes.
LORA_DROPOUT = 0.05  # Light regularisation dropout — this dataset is only
                      # 159 training examples, so some regularisation
                      # guards against memorising phrasing instead of the
                      # underlying facts.
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # All four
    # attention projections, not just q/v: with a dataset this small,
    # adapting the full attention block gives the model more surface area
    # to learn the domain's question-answering pattern from few examples.

# --- Training configuration ---
LEARNING_RATE = 2e-4  # Standard QLoRA learning rate for LoRA-only
                       # (frozen-base) fine-tuning; an order of magnitude
                       # higher than full fine-tuning because only the
                       # small adapter matrices are being updated.
BATCH_SIZE = 4  # Per-device batch size. Fits comfortably in 24GB VRAM at
                 # max_seq_length=1024 with 4-bit base weights; raise to 8
                 # only on a 40GB+ card (A40/A100).
GRAD_ACCUM = 4  # Effective batch size = BATCH_SIZE x GRAD_ACCUM = 16 —
                 # large enough for stable gradients on a 159-example
                 # dataset without needing a bigger per-step batch.
NUM_EPOCHS = 3  # Three passes over 159 examples is enough for a LoRA
                 # adapter to fit a narrow, structured domain; more risks
                 # overfitting a dataset this size.
MAX_SEQ_LEN = 1024  # Matches data_prep.py's DEFAULT_MAX_SEQ_LENGTH — the
                     # token-length check that gates the dataset assumes
                     # this exact ceiling, so it must not drift from it.

if not HF_TOKEN:
    raise SystemExit("Set your token first: export HF_TOKEN=hf_your_token_here")
login(token=HF_TOKEN)
print("Configuration loaded. HF login successful.")

# ------------------- 4-BIT QUANTISATION CONFIG (QLoRA) -------------------
# This is the "Q" in QLoRA: the frozen base model's weights are stored in
# 4 bits instead of 16/32, roughly quartering the memory an 8B model
# needs to just sit loaded — the difference between fitting on a single
# 24GB consumer GPU and needing a multi-GPU server. nf4 ("NormalFloat4")
# is a quantisation format tuned for how neural-network weights are
# actually distributed (roughly normal/bell-curve), so it loses less
# precision than a naive 4-bit format. Double quantisation additionally
# quantises the quantisation constants themselves, for a further small
# memory saving. bf16 compute dtype means the actual matrix math still
# happens at higher precision — only storage is compressed.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --------------------------- LOAD BASE MODEL -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)  # Gradient checkpointing + fp32 layer norms

# ---------------------------- LOAD TOKENISER -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --------------------------- LORA CONFIGURATION --------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)  # Freezes original weights

# The "LoRA" in QLoRA: this is the headline number that proves the method
# is working. Expect well under 1% of the 8B base model's parameters to
# be trainable — everything else stays frozen, which is *why* this fits
# on one consumer GPU and trains in ~15-30 minutes instead of the days a
# full fine-tune of every parameter would take.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})")

# ------------------------------ LOAD DATASET -----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/val.jsonl",
    },
)


def apply_chat_template(example):
    """Convert the messages list into the exact text string LLaMA was
    instruction-tuned on, including its special tokens. Training on any
    other format teaches the model the wrong input structure."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,  # False in TRAINING: the answer is already present
    )
    return {"text": text}


dataset = dataset.map(apply_chat_template)
print(f"Dataset prepared. Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")
print("Sample training text (first 300 chars):")
print(dataset["train"][0]["text"][:300])

# --------------------------- TRAINING ARGUMENTS ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",  # Gradually reduce LR as training progresses
    warmup_ratio=0.03,  # Warm up for 3% of steps before full LR — avoids a
                         # destabilising large first update against the
                         # frozen quantised base weights.
    weight_decay=0.001,  # Light L2 regularisation, consistent with the
                          # small LoRA dropout above — this dataset's size
                          # calls for mild, not aggressive, regularisation.
    bf16=True,  # bfloat16 training — safe on any Ampere-or-newer RunPod
                # GPU (the deployment guidance below requires one); would
                # need patching to fp16 only on pre-Ampere hardware.
    logging_steps=10,  # Log training loss every 10 steps
    eval_strategy="steps",
    eval_steps=10,  # Evaluate on the validation set every 10 steps — with
                     # ~10 steps/epoch at this batch/grad-accum setting,
                     # this checks validation loss roughly once per epoch.
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,  # Keep only the 3 most recent checkpoints
    load_best_model_at_end=True,  # Revert to the lowest-eval-loss checkpoint
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    optim="paged_adamw_8bit",  # Pages optimiser state to CPU RAM under
                                # memory pressure — needed headroom on a
                                # 24GB card training an 8B model in 4-bit.
)

# -------------------------------- TRAINER --------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,  # One example per sequence: no blurring between scenarios
)

# ----------------------------- LAUNCH TRAINING ---------------------------
steps_per_epoch = max(1, len(dataset["train"]) // (BATCH_SIZE * GRAD_ACCUM))
print("Starting training...")
print(f"Total optimiser steps: ~{steps_per_epoch * NUM_EPOCHS}")
print(f"Evaluating every {training_args.eval_steps} steps")
print("-" * 60)

trainer.train()

# ----------------------------- SAVE THE ADAPTER --------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# save_pretrained() only writes model/tokenizer files. trainer_state.json
# (the full loss/eval_loss log_history plot_loss.py needs) is written
# automatically inside each checkpoint-N/ subfolder during training, but
# NOT at this top level unless asked for explicitly — this call puts it
# at the path plot_loss.py and RUNPOD_GUIDE.md both expect.
trainer.save_state()
print(f"\nTraining complete. Adapter saved to: {OUTPUT_DIR}")
print(f"Copy {OUTPUT_DIR}/trainer_state.json back locally, then run "
      f"`python plot_loss.py` to generate the loss curve.")
