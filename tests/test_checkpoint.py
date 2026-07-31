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


def _make_trainer(tmp_path: Path, config: GPTConfig) -> Trainer:
    import numpy as np

    for name, n in (("train.bin", 2000), ("val.bin", 500)):
        ids = np.random.default_rng(0).integers(0, config.vocab_size, size=n).astype(np.uint16)
        ids.tofile(tmp_path / name)

    train_data = TokenDataset(tmp_path / "train.bin", config.block_size)
    val_data = TokenDataset(tmp_path / "val.bin", config.block_size)
    model = GPT(config)
    train_config = TrainConfig(out_dir=str(tmp_path / "run"), device="cpu", batch_size=8)
    return Trainer(model, config, train_config, train_data, val_data)


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
