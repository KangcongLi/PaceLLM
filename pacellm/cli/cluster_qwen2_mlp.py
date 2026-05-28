#!/usr/bin/env python3
"""Reorder Qwen2 FFN weights with Cortical Expert Clustering."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pacellm import CorticalClusteringConfig, cluster_and_rearrange_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Source Hugging Face model path or id.")
    parser.add_argument("--output-dir", required=True, help="Directory for the reordered model.")
    parser.add_argument("--n-experts", type=int, default=64, help="Number of equal-size cortical experts.")
    parser.add_argument("--backend", choices=["auto", "torch", "k_means_constrained"], default="auto")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for cached cluster labels.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    config = CorticalClusteringConfig(
        n_experts=args.n_experts,
        backend=args.backend,
        cache_dir=args.cache_dir,
    )
    reordered = cluster_and_rearrange_state_dict(model.state_dict(), config)
    model.load_state_dict(reordered)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved clustered model to {output_dir}")


if __name__ == "__main__":
    main()
