# lora-finetune-lab — Dev Notes for Claude Code

## Project intent

Lightweight LoRA fine-tuning toolkit.  Core library (data / config / eval / model_card) is
pure Python and always importable.  `train.py` defers all GPU imports inside functions — keep
it that way.

## Key constraints

- **No top-level torch/transformers/peft imports** anywhere except inside function bodies in
  `train.py`.
- All absolute imports (`from lora_lab.x import y`), never relative.
- `line-length = 100`, ruff selects E/F/I/UP/B, E501 ignored.
- Tests live in `tests/`, `src/` is on the pythonpath via `pyproject.toml`.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Running lint

```bash
ruff check src tests
```

## Adding a new metric

1. Add function to `src/lora_lab/eval.py`.
2. Add test to `tests/test_eval.py`.
3. Surface in `compare_models` return dict if it's an aggregate score.
4. Add column to `generate_model_card` table in `model_card.py`.

## Config changes

Edit `src/lora_lab/config.py` — keep `validate()` in sync with new fields.
`from_dict` / `to_dict` must also reflect any new fields.
