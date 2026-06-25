# Results — LoRA fine-tune, real before/after

A real LoRA fine-tune of **Qwen2.5-0.5B-Instruct** on AG-News topic
classification, measured before vs after on held-out test data.

## Setup
- **Base:** Qwen2.5-0.5B-Instruct (bf16)
- **Method:** LoRA — r=8, α=16, target `q_proj`/`v_proj`, dropout 0.05; 2 epochs, lr 2e-4
- **Trainable params:** 540,672 — **0.11%** of the 494M model
- **Task:** AG-News 4-way topic classification (World / Sports / Business / Sci-Tech),
  framed as generation + exact-match on the predicted label
- **Data:** 800 train / 250 test

## Result
| Model | Exact-match accuracy |
|---|---|
| base (zero-shot) | 0.532 |
| **LoRA fine-tuned** | **0.852** |
| **improvement** | **+0.320  (+32.0 points)** |

Training **0.11% of the parameters** lifted accuracy by **32 points** — the
"specialize a small model cheaply, and *prove* it improved" claim, measured on real data.

## Reproduce
GPU (Kaggle/Colab), ~10 min — one cell:
```bash
!wget -q -O finetune.py "https://raw.githubusercontent.com/im-girisankar/lora-finetune-lab/main/kaggle/finetune_kaggle.py" && python finetune.py
```
The script **checkpoints** — it caches the base-eval result and saves the trained
LoRA adapter, so a rerun skips the steps it already finished.

## Notes / honest limits
- Greedy decoding; exact-match on the label word. AG-News is an easy-ish 4-way task,
  chosen so the LoRA gain is unambiguous and fast to reproduce.
- bf16 LoRA (no quantization) — runs on any 16 GB GPU.
- A harder task (or QLoRA on a larger base) would be the natural next step.
