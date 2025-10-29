# Training

[Back to Docs Hub](README.md)

This guide explains how to fine-tune a causal language model on your curated dataset.

## Prerequisites

- Sufficient disk space and RAM; GPU recommended for larger models
- `transformers`, `accelerate`, `peft` installed (from requirements.txt)
- Training data at `./datasets/output.jsonl` or a curated JSONL

## Configuration

Training is controlled by `training/config.yaml`:

```yaml
model_name: "mistral"
output_dir: "./training/output"
per_device_train_batch_size: 1
learning_rate: 2e-5
num_train_epochs: 1
max_seq_length: 1024
lora:
  enabled: true
  r: 8
  alpha: 16
  dropout: 0.05
train_file: "./datasets/output.jsonl"
```

- `model_name`: HF identifier or local snapshot
- `output_dir`: where checkpoints will be saved
- LoRA parameters enable efficient fine-tuning

## Run training

CLI:

```bash
python -m cli train --config training/config.yaml
```

Python:

```python
from training.train import run_training
run_training("training/config.yaml")
```

The pipeline:

1. Loads JSONL into memory (simple reader to avoid package conflicts)
2. Tokenizes with `AutoTokenizer`
3. Formats examples as a simple prompt/response template
4. Trains with `Trainer`
5. Saves model and tokenizer to `output_dir`

## Customization

- Change the formatting in `training/train.py` (`_format_example`) to better suit your model
- Increase `num_train_epochs`, `batch_size` for more training (watch for OOM)
- Disable LoRA by setting `lora.enabled: false`

## Evaluation and prompt optimization

Use the learning utilities:

```bash
# Evaluate on a JSONL of {prompt,response}
python -m cli learn evaluate datasets/curated.test.jsonl

# Optimize system prompt from a list of candidates
python -m cli learn optimize-prompt prompts/candidates.txt datasets/curated.test.jsonl
```

This writes the best system prompt to `prompts/system_prompt.txt`, which `llm.py` will pick up automatically.

## Troubleshooting

- OOM: reduce `max_seq_length` and batch size; use LoRA and/or 8-bit weights
- Tokenizer errors: ensure model_name matches tokenizer availability
- Slow epochs: use smaller dataset or fewer epochs; consider gradient checkpointing (not enabled in this minimal harness)

