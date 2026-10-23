#!/usr/bin/env python
"""Generate text from a trained checkpoint.

The checkpoint stores its own model_config, so generation is always
consistent with the exact architecture it was trained with -- no separate
config file to keep in sync."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinylm.model import GPT, GPTConfig
from tinylm.tokenizer import ByteLevelBPETokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def load_model(checkpoint_path: Path, device: str) -> tuple[GPT, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    default_tokenizer = REPO_ROOT / "data_raw" / "tinyshakespeare" / "tokenizer.json"
    parser.add_argument("--tokenizer", type=Path, default=default_tokenizer)
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument(
        "--allowed-special", action="append", default=[], metavar="TOKEN",
        help="reserved tokenizer token to recognize in the prompt (repeatable)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--num-samples", type=positive_int, default=1,
        help="number of completions to generate in one batch",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="reduce logits for tokens already present in the sequence (must be >= 1)",
    )
    parser.add_argument(
        "--no-repeat-ngram-size", type=positive_int, default=None,
        help="prevent generation from repeating an n-gram of this size",
    )
    parser.add_argument(
        "--suppress-special", action="append", default=[], metavar="TOKEN",
        help="forbid a tokenizer special token during generation (repeatable)",
    )
    parser.add_argument(
        "--eos-token", type=str, default=None,
        help="stop after generating this tokenizer special token (e.g. <|endoftext|>)",
    )
    parser.add_argument(
        "--min-new-tokens", type=nonnegative_int, default=0,
        help="minimum tokens to generate before --eos-token can stop a sample",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--greedy", action="store_true",
        help="always choose the highest-logit eligible token instead of sampling",
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)
    if args.eos_token is not None and args.eos_token not in tokenizer.special_tokens:
        parser.error(f"unknown special token: {args.eos_token!r}")
    unknown_suppressed = set(args.suppress_special) - tokenizer.special_tokens.keys()
    if unknown_suppressed:
        parser.error(f"unknown special token: {min(unknown_suppressed)!r}")
    if args.min_new_tokens > args.max_new_tokens:
        parser.error("--min-new-tokens must not exceed --max-new-tokens")
    if args.min_new_tokens and args.eos_token is None:
        parser.error("--min-new-tokens requires --eos-token")
    eos_token_id = tokenizer.special_tokens.get(args.eos_token)
    try:
        prompt_ids = tokenizer.encode(args.prompt, allowed_special=args.allowed_special)
    except ValueError as error:
        parser.error(str(error))
    model, ckpt = load_model(args.checkpoint, args.device)
    if tokenizer.vocab_size != model.config.vocab_size:
        parser.error(
            "tokenizer vocabulary size "
            f"{tokenizer.vocab_size} does not match checkpoint vocabulary size "
            f"{model.config.vocab_size}"
        )
    generator = (
        torch.Generator(device=torch.device(args.device)).manual_seed(args.seed)
        if args.seed is not None
        else None
    )

    print(f"loaded checkpoint from step {ckpt['step']} (val_loss={ckpt['val_loss']:.4f})")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device).repeat(
        args.num_samples, 1
    )

    try:
        out_ids = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            suppress_token_ids=[
                tokenizer.special_tokens[token] for token in args.suppress_special
            ],
            do_sample=not args.greedy,
            generator=generator,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.num_samples == 1:
        print(tokenizer.decode(out_ids[0].tolist()))
    else:
        for sample, row in enumerate(out_ids, 1):
            print(f"--- sample {sample} ---")
            print(tokenizer.decode(row.tolist()))


if __name__ == "__main__":
    main()
