#!/usr/bin/env python
"""Evaluation utilities:
  eval          - exact validation loss/perplexity for one checkpoint over its full val set
  compare       - side-by-side comparison table across multiple run summary.json files
  benchmark-attn - manual vs fused scaled_dot_product_attention forward-pass throughput
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tinylm.data.dataset import TokenDataset
from tinylm.model import GPT, GPTConfig
from tinylm.model.attention import CausalSelfAttention

REPO_ROOT = Path(__file__).resolve().parent.parent


def cmd_eval(args: argparse.Namespace) -> None:
    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(ckpt["model_config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    val_data = TokenDataset(args.data_dir / "val.bin", config.block_size)
    num_batches = max(1, len(val_data) // (args.batch_size * config.block_size))

    total_loss = 0.0
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = val_data.get_batch(args.batch_size, device, generator)
            _, loss = model(x, y)
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20))
    result = {"val_loss": avg_loss, "val_perplexity": perplexity, "num_batches": num_batches}
    print(json.dumps(result, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    rows = []
    for run_dir in args.runs:
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.exists():
            print(f"skipping {run_dir}: no summary.json")
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "config": s["config_name"],
                "params": s["num_params_non_embedding"],
                "val_loss": round(s["best_val_loss"], 4),
                "perplexity": round(s["best_val_perplexity"], 3),
                "train_seconds": round(s["total_train_seconds"], 1),
                "device": s["device"],
            }
        )

    header = (
        f"{'config':<10} {'params':>12} {'val_loss':>10} "
        f"{'perplexity':>11} {'train_sec':>10} {'device':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['config']:<10} {r['params']:>12,} {r['val_loss']:>10} "
            f"{r['perplexity']:>11} {r['train_seconds']:>10} {r['device']:>8}"
        )


def cmd_benchmark_attn(args: argparse.Namespace) -> None:
    device = args.device
    config = GPTConfig(
        vocab_size=512,
        block_size=args.block_size,
        n_layer=1,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    x = torch.randn(args.batch_size, args.block_size, args.n_embd, device=device)

    results = {}
    for use_fused in (False, True):
        attn = CausalSelfAttention(config, use_fused=use_fused).to(device)
        attn.eval()

        with torch.no_grad():
            for _ in range(args.warmup):
                attn(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(args.iters):
                attn(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - t0

        key = "fused_sdpa" if use_fused else "manual"
        results[key] = {
            "total_seconds": elapsed,
            "iters_per_sec": args.iters / elapsed,
        }

    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data_raw" / "tinyshakespeare"
    )
    p_eval.add_argument("--batch-size", type=int, default=64)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p_eval.add_argument("--device", type=str, default=default_device)
    p_eval.set_defaults(func=cmd_eval)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("runs", nargs="+", help="run directories, e.g. runs/nano runs/tiny")
    p_compare.set_defaults(func=cmd_compare)

    p_bench = sub.add_parser("benchmark-attn")
    p_bench.add_argument("--batch-size", type=int, default=32)
    p_bench.add_argument("--block-size", type=int, default=256)
    p_bench.add_argument("--n-head", type=int, default=6)
    p_bench.add_argument("--n-embd", type=int, default=384)
    p_bench.add_argument("--iters", type=int, default=50)
    p_bench.add_argument("--warmup", type=int, default=10)
    default_device_bench = "cuda" if torch.cuda.is_available() else "cpu"
    p_bench.add_argument("--device", type=str, default=default_device_bench)
    p_bench.set_defaults(func=cmd_benchmark_attn)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
