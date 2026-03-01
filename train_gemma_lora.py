#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Gemma squat-coach data.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def load_jsonl_dataset(train_file, eval_file):
    return load_dataset(
        "json",
        data_files={"train": train_file, "eval": eval_file},
    )


def build_features(tokenizer, messages, max_length):
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_token_type_ids=True,
    )
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_token_type_ids=True,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    prompt_len = min(len(prompt_tokens["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if len(labels) < len(input_ids):
        labels.extend([-100] * (len(input_ids) - len(labels)))
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if "token_type_ids" in full_tokens:
        features["token_type_ids"] = full_tokens["token_type_ids"]
    return features


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        rows = []
        has_token_type_ids = any("token_type_ids" in feature for feature in features)
        for feature in features:
            row = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            if has_token_type_ids:
                row["token_type_ids"] = feature.get("token_type_ids", [0] * len(feature["input_ids"]))
            rows.append(row)
        batch = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            row = feature["labels"][:max_len]
            row = row + [-100] * (max_len - len(row))
            labels.append(row)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_jsonl_dataset(args.train_file, args.eval_file)

    tokenized = dataset.map(
        lambda row: build_features(tokenizer, row["messages"], args.max_length),
        remove_columns=dataset["train"].column_names,
    )
    collator = SupervisedDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with Path(args.output_dir, "run_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)


if __name__ == "__main__":
    main()
