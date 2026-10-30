#!/usr/bin/env python
"""Train a GPT model on a prepared token dataset using a named config
(configs/nano.json, tiny.json, small.json)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tinylm.data.dataset import TokenDataset
from tinylm.model import GPT, GPTConfig
from tinylm.training import TrainConfig, Trainer, initialize_model

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="name under configs/, e.g. 'tiny'")
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data_raw" / "tinyshakespeare"
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="resume an interrupted run")
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="initialize a new named run from compatible checkpoint weights",
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--use-fused-attn", action="store_true")
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="use CUDA float16 autocast and gradient scaling",
    )
    parser.add_argument(
        "--mixed-precision-dtype", choices=("float16", "bfloat16"),
        help="autocast dtype (requires --mixed-precision; default: float16)",
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="recompute transformer blocks during backpropagation to reduce activation memory",
    )
    parser.add_argument(
        "--label-smoothing", type=float,
        help="training-only label smoothing in [0, 1) (default: 0)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="micro-batches to average per optimizer step",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="stop after this many validation checks without improvement",
    )
    args = parser.parse_args()

    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")
    if args.init_from and not args.run_name:
        parser.error("--init-from requires --run-name")
    initial_run_dir = REPO_ROOT / "runs" / args.run_name if args.init_from else None
    if (
        initial_run_dir is not None
        and initial_run_dir.resolve() == args.init_from.resolve().parent
    ):
        parser.error("--init-from output must differ from the source checkpoint directory")
    if initial_run_dir and initial_run_dir.exists():
        parser.error("--init-from output run already exists")
    if args.mixed_precision_dtype is not None and not args.mixed_precision:
        parser.error("--mixed-precision-dtype requires --mixed-precision")
    if args.mixed_precision and (
        torch.device(args.device).type != "cuda" or not torch.cuda.is_available()
    ):
        parser.error("--mixed-precision requires an available CUDA device")
    if (
        args.mixed_precision
        and args.mixed_precision_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("bfloat16 mixed precision is not supported by this CUDA device")
    if args.grad_accum_steps is not None and args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be a positive integer")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be a positive integer")
    if args.label_smoothing is not None and (
        not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1
    ):
        parser.error("--label-smoothing must be finite and in [0, 1)")

    config_path = REPO_ROOT / "configs" / f"{args.config}.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))

    model_config = GPTConfig.from_dict({**raw_config["model"], "vocab_size": meta["vocab_size"]})
    resume_checkpoint = (
        torch.load(args.resume, map_location=args.device, weights_only=False)
        if args.resume
        else None
    )
    initial_checkpoint = (
        torch.load(args.init_from, map_location=args.device, weights_only=False)
        if args.init_from
        else None
    )
    if args.resume and args.run_name:
        parser.error("--run-name cannot be used with --resume")
    train_options = {**raw_config["train"]}
    if args.grad_accum_steps is not None:
        train_options["grad_accum_steps"] = args.grad_accum_steps
    if args.early_stopping_patience is not None:
        train_options["early_stopping_patience"] = args.early_stopping_patience
    if args.mixed_precision:
        train_options["mixed_precision"] = True
    if args.mixed_precision_dtype is not None:
        train_options["mixed_precision_dtype"] = args.mixed_precision_dtype
    if args.gradient_checkpointing:
        train_options["gradient_checkpointing"] = True
    if args.label_smoothing is not None:
        train_options["label_smoothing"] = args.label_smoothing
    train_config = TrainConfig(
        out_dir=str(
            args.resume.resolve().parent
            if args.resume
            else initial_run_dir or REPO_ROOT / "runs" / args.config
        ),
        device=args.device,
        **train_options,
    )

    torch.manual_seed(train_config.seed)
    train_data = TokenDataset(args.data_dir / "train.bin", model_config.block_size)
    val_data = TokenDataset(args.data_dir / "val.bin", model_config.block_size)

    model = GPT(model_config, use_fused_attn=args.use_fused_attn)
    if initial_checkpoint is not None:
        try:
            initialize_model(model, model_config, initial_checkpoint)
        except ValueError as error:
            parser.error(str(error))
    print(f"model params (non-embedding): {model.num_params():,}")
    print(f"device: {train_config.device}")

    try:
        trainer = Trainer(
            model,
            model_config,
            train_config,
            train_data,
            val_data,
            resume_checkpoint=resume_checkpoint,
        )
    except ValueError as error:
        parser.error(str(error))

    result = trainer.train()

    summary = {
        "config_name": args.config,
        "model_config": model_config.to_dict(),
        "train_config": train_options,
        "best_val_loss": result["best_val_loss"],
        "best_val_perplexity": math.exp(min(result["best_val_loss"], 20)),
        "num_params_non_embedding": model.num_params(),
        "total_train_seconds": result["elapsed_seconds"],
        "completed_step": result["completed_step"],
        "stopped_early": result["stopped_early"],
        "device": train_config.device,
    }
    out_dir = Path(train_config.out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
