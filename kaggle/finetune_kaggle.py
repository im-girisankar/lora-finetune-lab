# =====================================================================
# lora-finetune-lab — a REAL LoRA fine-tune with before/after eval.
# Copy-paste one cell into Kaggle/Colab (GPU runtime, internet ON).
# =====================================================================
# Fine-tunes Qwen2.5-0.5B-Instruct with LoRA on AG-News topic classification
# (4 classes) and reports exact-match accuracy BEFORE vs AFTER training, plus a
# generated MODEL_CARD. bf16 (no bitsandbytes) so it runs anywhere with 16 GB.
# ~8-12 min on a T4. Proves the "specialize a small model and measure the gain" claim.
# =====================================================================

import subprocess, sys
def pip(*p): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], check=True)
pip("transformers>=4.46", "peft>=0.11", "datasets>=2.19", "accelerate>=0.30")

import numpy as np, torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN, N_TEST = 800, 250
SEED = 7
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
PROMPT = ("Classify the topic of this news text as exactly one of: "
          "World, Sports, Business, Sci/Tech.\nText: {text}\nTopic:")
torch.manual_seed(SEED)

print("Loading AG-News ...")
ds = load_dataset("fancyzhx/ag_news")
tr = ds["train"].shuffle(seed=SEED).select(range(N_TRAIN))
te = ds["test"].shuffle(seed=SEED).select(range(N_TEST))

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
DEV = next(model.parameters()).device


def chat(text, label=None):
    msgs = [{"role": "user", "content": PROMPT.format(text=text)}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s + (" " + label + tok.eos_token if label is not None else "")


@torch.no_grad()
def evaluate(split, tag=""):
    model.eval(); correct = 0; n = len(split)
    for i, row in enumerate(split):
        enc = tok(chat(row["text"]), return_tensors="pt").to(DEV)
        out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = next((lab for lab in LABELS if lab.lower() in gen.lower()), "")
        correct += int(pred == LABELS[row["label"]])
        if (i + 1) % 50 == 0:
            print(f"    {tag} {i+1}/{n}  (running acc {correct/(i+1):.3f})")
    return correct / n


print("Evaluating BASE model (before fine-tuning) ...")
acc_before = evaluate(te, "base")
print(f"  base accuracy: {acc_before:.3f}")

# ---- LoRA fine-tune ----
print("Preparing LoRA fine-tune ...")
texts = [chat(r["text"], LABELS[r["label"]]) for r in tr]
def tok_fn(b):
    enc = tok(b["text"], truncation=True, max_length=160, padding="max_length")
    enc["labels"] = [[(t if t != tok.pad_token_id else -100) for t in ids] for ids in enc["input_ids"]]
    return enc
train_ds = Dataset.from_dict({"text": texts}).map(tok_fn, batched=True, remove_columns=["text"])

model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()

Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="lora_out", per_device_train_batch_size=8, num_train_epochs=2,
        learning_rate=2e-4, bf16=True, logging_steps=25, save_strategy="no",
        report_to="none", seed=SEED),
    train_dataset=train_ds, data_collator=default_data_collator,
).train()

print("Evaluating TUNED model (after fine-tuning) ...")
acc_after = evaluate(te, "tuned")

# ---- report + model card ----
delta = acc_after - acc_before
print("\n========= lora-finetune-lab : REAL before/after =========")
print(f"model={MODEL_ID}  task=AG-News (4-way)  N_train={N_TRAIN}  N_test={N_TEST}")
print(f"  base (no fine-tune): {acc_before:.3f}")
print(f"  LoRA fine-tuned    : {acc_after:.3f}")
print(f"  improvement        : {delta:+.3f}  ({delta*100:+.1f} points)")

card = f"""# Model Card — Qwen2.5-0.5B + LoRA (AG-News)

- **Base model:** {MODEL_ID}
- **Method:** LoRA (r=8, alpha=16, target q_proj/v_proj), 2 epochs, lr 2e-4, bf16
- **Task:** AG-News topic classification (World / Sports / Business / Sci-Tech), generation + exact match
- **Data:** {N_TRAIN} train / {N_TEST} test

## Result (exact-match accuracy)
| Model | Accuracy |
|---|---|
| base (zero-shot) | {acc_before:.3f} |
| **LoRA fine-tuned** | **{acc_after:.3f}** |
| improvement | **{delta:+.3f}** ({delta*100:+.1f} pts) |
"""
open("MODEL_CARD_generated.md", "w", encoding="utf-8").write(card)
print("\nSaved MODEL_CARD_generated.md")
