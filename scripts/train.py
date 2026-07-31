#!/usr/bin/env python
"""Train a GPT model on a prepared token dataset using a named config
(configs/nano.json, tiny.json, small.json)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tinylm.data.dataset import TokenDataset
from tinylm.model import GPT, GPTConfig
from tinylm.training import TrainConfig, Trainer

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="name under configs/, e.g. 'tiny'")
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data_raw" / "tinyshakespeare"
    )
    parser.add_argument("--run-name", type=str, default=None)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--use-fused-attn", action="store_true")
    args = parser.parse_args()

    config_path = REPO_ROOT / "configs" / f"{args.config}.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))

    model_config = GPTConfig.from_dict({**raw_config["model"], "vocab_size": meta["vocab_size"]})
    train_config = TrainConfig(
        out_dir=str(REPO_ROOT / "runs" / (args.run_name or args.config)),
        device=args.device,
        **raw_config["train"],
    )

    torch.manual_seed(train_config.seed)
    train_data = TokenDataset(args.data_dir / "train.bin", model_config.block_size)
    val_data = TokenDataset(args.data_dir / "val.bin", model_config.block_size)

    model = GPT(model_config, use_fused_attn=args.use_fused_attn)
    print(f"model params (non-embedding): {model.num_params():,}")
    print(f"device: {train_config.device}")

    trainer = Trainer(model, model_config, train_config, train_data, val_data)

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    summary = {
        "config_name": args.config,
        "model_config": model_config.to_dict(),
        "train_config": raw_config["train"],
        "best_val_loss": result["best_val_loss"],
        "best_val_perplexity": math.exp(min(result["best_val_loss"], 20)),
        "num_params_non_embedding": model.num_params(),
        "total_train_seconds": elapsed,
        "device": train_config.device,
    }
    out_dir = Path(train_config.out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
