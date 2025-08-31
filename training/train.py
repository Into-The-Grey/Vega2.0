"""
train.py - Simple fine-tuning harness using Hugging Face + Accelerate + optional LoRA/PEFT

Notes:
- This is a minimal reference pipeline for local experiments.
- Expects datasets/output.jsonl with {"prompt","response"} lines.
- Uses a causal LM; adjust for your chosen model.
- For GPU, ensure correct CUDA wheels (bitsandbytes optional).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None

    def get_peft_model(model, config):
        return model


@dataclass
class TrainConfig:
    model_name: str
    output_dir: str
    per_device_train_batch_size: int
    learning_rate: float
    num_train_epochs: int
    max_seq_length: int
    lora_enabled: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    train_file: str


def _read_yaml(path: str) -> Dict:
    import yaml  # pyyaml is included transitively with datasets; if missing, install

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config(path: str) -> TrainConfig:
    d = _read_yaml(path)
    lc = d.get("lora", {})
    return TrainConfig(
        model_name=d["model_name"],
        output_dir=d["output_dir"],
        per_device_train_batch_size=int(d.get("per_device_train_batch_size", 1)),
        learning_rate=float(d.get("learning_rate", 2e-5)),
        num_train_epochs=int(d.get("num_train_epochs", 1)),
        max_seq_length=int(d.get("max_seq_length", 1024)),
        lora_enabled=bool(lc.get("enabled", False)),
        lora_r=int(lc.get("r", 8)),
        lora_alpha=int(lc.get("alpha", 16)),
        lora_dropout=float(lc.get("dropout", 0.05)),
        train_file=d.get("train_file", "./datasets/output.jsonl"),
    )


def _format_example(ex: Dict, tokenizer, max_len: int):
    # Simple prompt-response concatenation with EOS
    text = f"<s>Prompt: {ex['prompt']}\nResponse: {ex['response']}\n"  # naive format
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors=None,
    )
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_training(config_path: str):
    cfg = _load_config(config_path)

    # Load dataset from JSONL manually to avoid package name collision
    raw = list(_read_jsonl(cfg.train_file))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        return _format_example(example, tokenizer, cfg.max_seq_length)

    tokenized = [preprocess(ex) for ex in raw]

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    # Optional LoRA
    if cfg.lora_enabled and LoraConfig is not None:
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=False,
        bf16=False,
        report_to=[],
    )

    class JsonlDataset:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    trainer = Trainer(model=model, args=args, train_dataset=JsonlDataset(tokenized))

    trainer.train()

    # Save final model/tokenizer
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "training/config.yaml"
    run_training(path)
