#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = "You are a squat coach. Given structured pose-derived squat features, return a short coaching cue only. Keep it concise and actionable."


def build_user_prompt(row):
    payload = {
        "task": "squat_coaching",
        "input": row["input"],
        "rules": [
            "Return one short coaching cue.",
            "Do not explain the reasoning.",
            "Do not mention raw variable names.",
            "If is_setup is true, give a setup/framing instruction.",
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def normalize(text):
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()


def overlap(expected, actual):
    left = set(normalize(expected))
    right = set(normalize(actual))
    return len(left & right) / max(1, len(left))


def quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(model_id, hf_token):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config(),
        attn_implementation="sdpa",
    )


def run_variant(name, model, tokenizer, rows):
    model.eval()
    results = []
    for idx, row in enumerate(rows, start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append(
            {
                "id": row["id"],
                "label": row.get("label"),
                "expected_feedback": row["expected_feedback"],
                "predicted_feedback": text,
                "token_overlap": round(overlap(row["expected_feedback"], text), 4),
                "exact_match": row["expected_feedback"].strip().lower() == text.strip().lower(),
            }
        )
        if idx % 10 == 0:
            print(f"{name}: {idx}/{len(rows)}", flush=True)

    avg = sum(r["token_overlap"] for r in results) / max(1, len(results))
    exact = sum(r["exact_match"] for r in results) / max(1, len(results))
    return {
        "variant": name,
        "avg_token_overlap": round(avg, 4),
        "exact_match_rate": round(exact, 4),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.eval_path) if line.strip()]
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    summary = {
        "name": args.name,
        "model_id": args.model_id,
        "dataset": args.eval_path,
        "cases": len(rows),
        "variants": [],
    }

    base = load_base_model(args.model_id, hf_token)
    summary["variants"].append(run_variant("base", base, tokenizer, rows))
    print("SUMMARY", json.dumps(summary["variants"][-1]), flush=True)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapted_base = load_base_model(args.model_id, hf_token)
    adapter = PeftModel.from_pretrained(adapted_base, args.adapter_dir)
    summary["variants"].append(run_variant("adapter", adapter, tokenizer, rows))
    print("SUMMARY", json.dumps(summary["variants"][-1]), flush=True)
    del adapter
    del adapted_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
