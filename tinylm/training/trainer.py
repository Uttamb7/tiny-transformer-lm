from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
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
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20
    seed: int = 1337
    device: str = "cuda"


class Trainer:
    def __init__(
        self,
        model: GPT,
        model_config: GPTConfig,
        train_config: TrainConfig,
        train_data: TokenDataset,
        val_data: TokenDataset,
    ) -> None:
        self.model = model.to(train_config.device)
        self.model_config = model_config
        self.train_config = train_config
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = self._build_optimizer()
        self.generator = torch.Generator().manual_seed(train_config.seed)
        self.metrics: list[dict] = []

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

    def save_checkpoint(self, path: Path, step: int, val_loss: float, best_val_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "model_config": self.model_config.to_dict(),
                "step": step,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
            path,
        )

    def train(self) -> dict:
        cfg = self.train_config
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.jsonl"

        best_val_loss = float("inf")
        t0 = time.time()
        tokens_per_step = cfg.batch_size * self.model_config.block_size

        self.model.train()
        for step in range(cfg.max_steps + 1):
            lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr_ratio)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            if step % cfg.eval_interval == 0 or step == cfg.max_steps:
                losses = self.estimate_loss()
                elapsed = time.time() - t0
                perplexity = math.exp(min(losses["val"], 20))
                record = {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "val_perplexity": perplexity,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "tokens_per_sec": (step * tokens_per_step / elapsed) if step > 0 else 0.0,
                }
                self.metrics.append(record)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    self.save_checkpoint(out_dir / "best.pt", step, losses["val"], best_val_loss)
                self.save_checkpoint(out_dir / "last.pt", step, losses["val"], best_val_loss)

            if step == cfg.max_steps:
                break

            x, y = self.train_data.get_batch(cfg.batch_size, cfg.device, self.generator)
            _, loss = self.model(x, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"training diverged at step {step}: loss={loss.item()}. "
                    "Try a lower max_lr or increase warmup_steps."
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

        return {"best_val_loss": best_val_loss, "metrics": self.metrics}
