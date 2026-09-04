import json
from pathlib import Path

import pytest
import torch

from tinylm.data.dataset import TokenDataset
from tinylm.model import GPT, GPTConfig
from tinylm.training import TrainConfig, Trainer


def make_config(**overrides) -> GPTConfig:
    base = dict(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=8, dropout=0.0)
    base.update(overrides)
    return GPTConfig(**base)


def _make_trainer(
    tmp_path: Path,
    config: GPTConfig,
    run_name: str = "run",
    resume_checkpoint: dict | None = None,
    **train_overrides,
) -> Trainer:
    import numpy as np

    for name, n in (("train.bin", 2000), ("val.bin", 500)):
        ids = np.random.default_rng(0).integers(0, config.vocab_size, size=n).astype(np.uint16)
        ids.tofile(tmp_path / name)

    train_data = TokenDataset(tmp_path / "train.bin", config.block_size)
    val_data = TokenDataset(tmp_path / "val.bin", config.block_size)
    model = GPT(config)
    train_options = {
        "out_dir": str(tmp_path / run_name),
        "device": "cpu",
        "batch_size": 8,
    }
    train_options.update(train_overrides)
    train_config = TrainConfig(**train_options)
    return Trainer(
        model,
        config,
        train_config,
        train_data,
        val_data,
        resume_checkpoint=resume_checkpoint,
    )


def test_checkpoint_round_trip_preserves_weights_and_outputs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = make_config()
    trainer = _make_trainer(tmp_path, config)

    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt_path, step=5, val_loss=1.23, best_val_loss=1.23)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["step"] == 5
    assert loaded["val_loss"] == 1.23
    assert loaded["model_config"] == config.to_dict()

    restored_model = GPT(GPTConfig.from_dict(loaded["model_config"]))
    restored_model.load_state_dict(loaded["model_state"])

    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    trainer.model.eval()
    restored_model.eval()
    with torch.no_grad():
        logits_a, _ = trainer.model(x)
        logits_b, _ = restored_model(x)
    assert torch.equal(logits_a, logits_b)


def test_loading_checkpoint_with_wrong_architecture_raises(tmp_path: Path) -> None:
    config = make_config()
    trainer = _make_trainer(tmp_path, config)
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(ckpt_path, step=1, val_loss=2.0, best_val_loss=2.0)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mismatched_config = GPTConfig.from_dict({**loaded["model_config"], "n_embd": 16, "n_head": 2})
    mismatched_model = GPT(mismatched_config)

    with pytest.raises(RuntimeError):
        mismatched_model.load_state_dict(loaded["model_state"])


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    class Interrupted(Exception):
        pass

    config = make_config(dropout=0.2)
    train_options = {
        "max_steps": 4,
        "warmup_steps": 1,
        "max_lr": 1e-3,
        "eval_interval": 2,
        "eval_iters": 2,
        "seed": 7,
    }

    torch.manual_seed(11)
    uninterrupted = _make_trainer(tmp_path, config, "full", **train_options)
    uninterrupted.train()

    torch.manual_seed(11)
    interrupted = _make_trainer(tmp_path, config, "resumed", **train_options)
    save_checkpoint = interrupted.save_checkpoint

    def save_then_stop(path, step, val_loss, best_val_loss, elapsed_seconds=0.0):
        save_checkpoint(path, step, val_loss, best_val_loss, elapsed_seconds)
        if path.name == "last.pt" and step == 2:
            raise Interrupted

    interrupted.save_checkpoint = save_then_stop
    with pytest.raises(Interrupted):
        interrupted.train()

    checkpoint = torch.load(
        tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=False
    )
    torch.manual_seed(999)
    resumed = _make_trainer(
        tmp_path, config, "resumed", resume_checkpoint=checkpoint, **train_options
    )
    resumed.train()

    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], expected), name
    full_steps = [
        record["step"]
        for record in map(
            json.loads, (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()
        )
    ]
    resumed_steps = [
        record["step"]
        for record in map(
            json.loads, (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
        )
    ]
    assert resumed_steps == full_steps == [0, 2, 4]
    assert resumed.best_val_loss == uninterrupted.best_val_loss
    full_checkpoint = torch.load(
        tmp_path / "full" / "last.pt", map_location="cpu", weights_only=False
    )
    resumed_checkpoint = torch.load(
        tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=False
    )
    assert torch.equal(resumed_checkpoint["torch_rng_state"], full_checkpoint["torch_rng_state"])
    assert torch.equal(
        resumed_checkpoint["trainer_rng_state"], full_checkpoint["trainer_rng_state"]
    )
    assert resumed_checkpoint["optimizer_state"]["param_groups"] == full_checkpoint[
        "optimizer_state"
    ]["param_groups"]
    for parameter, expected_state in full_checkpoint["optimizer_state"]["state"].items():
        for name, expected in expected_state.items():
            assert torch.equal(
                resumed_checkpoint["optimizer_state"]["state"][parameter][name], expected
            )


def test_resume_rejects_incomplete_and_incompatible_checkpoints(tmp_path: Path) -> None:
    config = make_config()
    trainer = _make_trainer(tmp_path, config)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, step=1, val_loss=2.0, best_val_loss=2.0)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    incomplete = {key: value for key, value in checkpoint.items() if key != "trainer_rng_state"}
    with pytest.raises(ValueError, match="missing trainer_rng_state"):
        _make_trainer(tmp_path, config, "incomplete", resume_checkpoint=incomplete)

    with pytest.raises(ValueError, match="training configuration"):
        _make_trainer(
            tmp_path,
            config,
            "mismatch",
            resume_checkpoint=checkpoint,
            max_lr=9e-4,
        )
