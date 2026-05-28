#!/usr/bin/env python3
"""Minimal example for enabling AMB on selected Qwen2 MLP layers."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pacellm import ActivationMemoryConfig, enable_amb_for_qwen2, reset_amb_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 26])
    parser.add_argument("--prompt", default="Summarize the following long document: ...")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    memory_config = ActivationMemoryConfig(bank_size=100, theta_high=0.75, theta_low=0.25)
    enable_amb_for_qwen2(model, layers=args.layers, memory_config=memory_config)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        reset_amb_memory(model)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
