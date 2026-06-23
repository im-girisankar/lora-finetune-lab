# lora-finetune-lab

A self-contained Python toolkit for LoRA fine-tuning experiments on causal language models.
The core library is pure Python — no GPU needed to prep data, run evaluations on pre-computed
outputs, or generate model cards.  The actual training path is deliberately gated behind an
optional extra so the repo remains lightweight for exploration and CI.

---

## Why this exists

Fine-tuning a language model is cheap to claim and hard to prove.  This lab provides the
scaffolding to **measure improvement** — exact match and token-F1 before and after — and
auto-generates a model card that shows the numbers honestly.  The PM angle: every training
run produces evidence, not just a checkpoint.

---

## Quickstart

```bash
# 1. Install (pure Python, no GPU required for data / eval / card)
pip install -e .

# 2. Inspect your data
loralab prep --data examples/data.jsonl

# 3. Run a stub evaluation (demonstrates the pipeline)
loralab eval --data examples/data.jsonl --model-name my-baseline

# 4. Generate a model card
loralab card --data examples/data.jsonl --config config.json --output MODEL_CARD.md

# 5. Train (requires GPU + train extra)
pip install -e ".[train]"
loralab train --data examples/data.jsonl --config config.json
```

---

## Project layout

```
src/lora_lab/
    __init__.py      — version
    data.py          — JSONL loading, prompt formatting, splitting, stats
    config.py        — LoraLabConfig dataclass + validation
    eval.py          — exact_match, token_f1, run_eval, compare_models
    model_card.py    — markdown model card generator
    train.py         — LoRA training (lazy torch/transformers imports)
    cli.py           — loralab CLI (prep / eval / card / train)

tests/               — 20+ unit tests, no torch dependency
examples/
    data.jsonl       — 10 realistic instruction-tuning examples
```

---

## Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | Done | Core library: data, config, eval, model card |
| M2 | Done | CLI with prep / eval / card / train subcommands |
| M3 | Planned | End-to-end training on TinyLlama with real GPU run + logged results |
| M4 | Planned | Weights & Biases integration + sweep over rank/alpha configs |

---

## Training note

Training requires a CUDA-capable GPU and the `train` extra:

```bash
pip install -e ".[train]"
# torch, transformers, peft, datasets are pulled in
```

Without the extra, all other commands (`prep`, `eval`, `card`) work fine.

---

## Development

```bash
pip install -e ".[dev]"
pytest          # run tests
ruff check src  # lint
ruff format src # format
```

---

## License

MIT — see [LICENSE](LICENSE).
