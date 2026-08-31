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
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--eos-token", type=str, default=None,
        help="stop after generating this tokenizer special token (e.g. <|endoftext|>)",
    )
    parser.add_argument("--seed", type=int, default=None)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer)
    if args.eos_token is not None and args.eos_token not in tokenizer.special_tokens:
        parser.error(f"unknown special token: {args.eos_token!r}")
    eos_token_id = tokenizer.special_tokens.get(args.eos_token)
    model, ckpt = load_model(args.checkpoint, args.device)

    print(f"loaded checkpoint from step {ckpt['step']} (val_loss={ckpt['val_loss']:.4f})")

    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    out_ids = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )
    print(tokenizer.decode(out_ids[0].tolist()))


if __name__ == "__main__":
    main()
