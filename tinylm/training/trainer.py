from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tinylm.data.dataset import TokenDataset
from tinylm.model import GPT, GPTConfig
from tinylm.training.lr_schedule import get_lr


@dataclass
class TrainConfig:
    out_dir: str
    max_steps: int = 2000
    warmup_steps: int = 100
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 50
    early_stopping_patience: int | None = None
    log_interval: int = 20
    seed: int = 1337
    device: str = "cuda"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.grad_accum_steps, int)
            or isinstance(self.grad_accum_steps, bool)
            or self.grad_accum_steps < 1
        ):
            raise ValueError("grad_accum_steps must be a positive integer")
        if self.early_stopping_patience is not None and (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be a positive integer")


class Trainer:
    def __init__(
        self,
        model: GPT,
        model_config: GPTConfig,
        train_config: TrainConfig,
        train_data: TokenDataset,
        val_data: TokenDataset,
        resume_checkpoint: dict | None = None,
    ) -> None:
        self.model = model.to(train_config.device)
        self.model_config = model_config
        self.train_config = train_config
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = self._build_optimizer()
        self.generator = torch.Generator().manual_seed(train_config.seed)
        self.metrics: list[dict] = []
        self.resume_step = 0
        self.best_val_loss = float("inf")
        self.elapsed_seconds = 0.0
        self.no_improvement_evals = 0
        self.stopped_early = False
        self.resumed = False
        if resume_checkpoint is not None:
            self.restore_checkpoint(resume_checkpoint)

    def _resume_config(self) -> dict:
        config = asdict(self.train_config)
        del config["out_dir"]
        del config["device"]
        return config

    def restore_checkpoint(self, checkpoint: dict) -> None:
        required = {
            "model_state", "optimizer_state", "model_config", "train_config", "step",
            "best_val_loss", "torch_rng_state", "trainer_rng_state", "device_type",
            "use_fused_attn", "elapsed_seconds",
        }
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError("checkpoint cannot be resumed; missing " + ", ".join(missing))
        if checkpoint["model_config"] != self.model_config.to_dict():
            raise ValueError("checkpoint model configuration does not match the requested config")
        checkpoint_train_config = {**checkpoint["train_config"]}
        checkpoint_train_config.setdefault("grad_accum_steps", 1)
        checkpoint_train_config.setdefault("early_stopping_patience", None)
        if checkpoint_train_config != self._resume_config():
            raise ValueError(
                "checkpoint training configuration does not match the requested config"
            )
        if checkpoint["device_type"] != torch.device(self.train_config.device).type:
            raise ValueError("checkpoint device type does not match the requested device")
        if checkpoint["use_fused_attn"] != self.model.use_fused_attn:
            raise ValueError(
                "checkpoint attention implementation does not match the requested model"
            )
        step = checkpoint["step"]
        if not isinstance(step, int) or step < 0 or step > self.train_config.max_steps:
            raise ValueError("checkpoint step is outside the requested training schedule")

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        self.generator.set_state(checkpoint["trainer_rng_state"].cpu())
        if checkpoint["device_type"] == "cuda":
            cuda_state = checkpoint.get("cuda_rng_state")
            if not cuda_state:
                raise ValueError("checkpoint cannot be resumed; missing cuda_rng_state")
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
        self.resume_step = step
        self.best_val_loss = float(checkpoint["best_val_loss"])
        self.elapsed_seconds = float(checkpoint["elapsed_seconds"])
        self.no_improvement_evals = int(checkpoint.get("no_improvement_evals", 0))
        self.stopped_early = bool(checkpoint.get("stopped_early", False))
        self.resumed = True

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.train_config.max_lr, betas=(0.9, 0.95))

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        out: dict[str, float] = {}
        for split, dataset in (("train", self.train_data), ("val", self.val_data)):
            losses = torch.zeros(self.train_config.eval_iters)
            for i in range(self.train_config.eval_iters):
                x, y = dataset.get_batch(
                    self.train_config.batch_size, self.train_config.device, self.generator
                )
                _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(
        self,
        path: Path,
        step: int,
        val_loss: float,
        best_val_loss: float,
        elapsed_seconds: float = 0.0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "model_config": self.model_config.to_dict(),
                "train_config": self._resume_config(),
                "step": step,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "elapsed_seconds": elapsed_seconds,
                "no_improvement_evals": self.no_improvement_evals,
                "stopped_early": self.stopped_early,
                "torch_rng_state": torch.get_rng_state(),
                "trainer_rng_state": self.generator.get_state(),
                "device_type": torch.device(self.train_config.device).type,
                "use_fused_attn": self.model.use_fused_attn,
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if torch.device(self.train_config.device).type == "cuda"
                else None,
            },
            path,
        )

    def train(self) -> dict:
        cfg = self.train_config
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.jsonl"

        best_val_loss = self.best_val_loss
        if self.resumed:
            if not metrics_path.exists():
                raise ValueError("cannot resume without the existing metrics.jsonl")
            records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            if not records or records[-1].get("step") != self.resume_step:
                raise ValueError("metrics.jsonl does not end at the checkpoint step")
            if self.stopped_early:
                return {
                    "best_val_loss": best_val_loss,
                    "metrics": self.metrics,
                    "elapsed_seconds": self.elapsed_seconds,
                    "completed_step": self.resume_step,
                    "stopped_early": True,
                }
        t0 = time.time()
        tokens_per_step = (
            cfg.batch_size * self.model_config.block_size * cfg.grad_accum_steps
        )

        self.model.train()
        completed_step = self.resume_step
        for step in range(self.resume_step, cfg.max_steps + 1):
            completed_step = step
            lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr_ratio)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            if not (self.resumed and step == self.resume_step) and (
                step % cfg.eval_interval == 0 or step == cfg.max_steps
            ):
                losses = self.estimate_loss()
                elapsed = self.elapsed_seconds + time.time() - t0
                perplexity = math.exp(min(losses["val"], 20))
                record = {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "val_perplexity": perplexity,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "tokens_per_sec": step * tokens_per_step / elapsed if step > 0 else 0.0,
                }
                self.metrics.append(record)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    self.no_improvement_evals = 0
                    self.save_checkpoint(
                        out_dir / "best.pt", step, losses["val"], best_val_loss, elapsed
                    )
                else:
                    self.no_improvement_evals += 1
                self.stopped_early = (
                    cfg.early_stopping_patience is not None
                    and self.no_improvement_evals >= cfg.early_stopping_patience
                )
                self.save_checkpoint(
                    out_dir / "last.pt", step, losses["val"], best_val_loss, elapsed
                )
                if self.stopped_early:
                    break

            if step == cfg.max_steps:
                break

            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum_steps):
                x, y = self.train_data.get_batch(cfg.batch_size, cfg.device, self.generator)
                _, loss = self.model(x, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"training diverged at step {step}: loss={loss.item()}. "
                        "Try a lower max_lr or increase warmup_steps."
                    )
                (loss / cfg.grad_accum_steps).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

        self.best_val_loss = best_val_loss
        self.elapsed_seconds += time.time() - t0
        return {
            "best_val_loss": best_val_loss,
            "metrics": self.metrics,
            "elapsed_seconds": self.elapsed_seconds,
            "completed_step": completed_step,
            "stopped_early": self.stopped_early,
        }
