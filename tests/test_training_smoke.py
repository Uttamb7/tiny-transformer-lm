from pathlib import Path

import numpy as np
import pytest
import torch

from tinylm.data.dataset import TokenDataset
from tinylm.model import GPT, GPTConfig
from tinylm.training import TrainConfig, Trainer


def make_synthetic_bin(
    path: Path, vocab_size: int, num_tokens: int, pattern_len: int, seed: int = 0
) -> None:
    """A repeating pattern is trivially learnable, so a tiny model's loss
    should clearly drop within a handful of steps -- a real, checkable signal
    rather than just 'it ran without crashing'."""
    rng = np.random.default_rng(seed)
    pattern = rng.integers(0, vocab_size, size=pattern_len)
    reps = num_tokens // pattern_len + 1
    ids = np.tile(pattern, reps)[:num_tokens].astype(np.uint16)
    ids.tofile(path)


def test_training_reduces_loss_on_repetitive_pattern(tmp_path: Path) -> None:
    torch.manual_seed(0)
    vocab_size = 32
    block_size = 16

    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    make_synthetic_bin(train_path, vocab_size, num_tokens=4000, pattern_len=block_size, seed=0)
    make_synthetic_bin(val_path, vocab_size, num_tokens=1000, pattern_len=block_size, seed=0)

    train_data = TokenDataset(train_path, block_size)
    val_data = TokenDataset(val_path, block_size)

    model_config = GPTConfig(
        vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )
    model = GPT(model_config)

    train_config = TrainConfig(
        out_dir=str(tmp_path / "run"),
        max_steps=150,
        warmup_steps=10,
        batch_size=16,
        max_lr=3e-3,
        eval_interval=25,
        eval_iters=5,
        device="cpu",
    )

    trainer = Trainer(model, model_config, train_config, train_data, val_data)
    result = trainer.train()

    metrics = result["metrics"]
    assert len(metrics) >= 2
    assert all(m["train_loss"] == m["train_loss"] for m in metrics)  # NaN != NaN
    assert metrics[-1]["train_loss"] < metrics[0]["train_loss"]

    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "metrics.jsonl").exists()


def test_diverging_loss_raises_clear_error(tmp_path: Path) -> None:
    vocab_size = 32
    block_size = 8
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    make_synthetic_bin(train_path, vocab_size, num_tokens=2000, pattern_len=block_size, seed=0)
    make_synthetic_bin(val_path, vocab_size, num_tokens=500, pattern_len=block_size, seed=0)

    train_data = TokenDataset(train_path, block_size)
    val_data = TokenDataset(val_path, block_size)
    model_config = GPTConfig(
        vocab_size=vocab_size, block_size=block_size, n_layer=1, n_head=1, n_embd=8, dropout=0.0
    )
    model = GPT(model_config)

    train_config = TrainConfig(
        out_dir=str(tmp_path / "run_diverge"),
        max_steps=20,
        warmup_steps=1,
        batch_size=8,
        max_lr=1e6,
        eval_interval=100,
        eval_iters=2,
        device="cpu",
    )
    trainer = Trainer(model, model_config, train_config, train_data, val_data)

    with pytest.raises(RuntimeError, match="diverged"):
        trainer.train()


def test_gradient_accumulation_uses_micro_batches_and_one_update_per_step(
    tmp_path: Path,
) -> None:
    vocab_size = 16
    block_size = 4
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    make_synthetic_bin(train_path, vocab_size, 200, block_size)
    make_synthetic_bin(val_path, vocab_size, 100, block_size)
    train_data = TokenDataset(train_path, block_size)
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    trainer = Trainer(
        GPT(config),
        config,
        TrainConfig(
            out_dir=str(tmp_path / "accum"),
            max_steps=2,
            batch_size=2,
            grad_accum_steps=3,
            eval_interval=1,
            eval_iters=1,
            device="cpu",
        ),
        train_data,
        TokenDataset(val_path, block_size),
    )
    batch_calls = 0
    optimizer_steps = 0
    get_batch = train_data.get_batch
    optimizer_step = trainer.optimizer.step

    def counted_batch(*args, **kwargs):
        nonlocal batch_calls
        batch_calls += 1
        return get_batch(*args, **kwargs)

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return optimizer_step(*args, **kwargs)

    train_data.get_batch = counted_batch
    trainer.optimizer.step = counted_step
    trainer.estimate_loss = lambda: {"train": 1.0, "val": 1.0}
    result = trainer.train()

    assert batch_calls == 6
    assert optimizer_steps == 2
    final_metric = result["metrics"][-1]
    assert final_metric["tokens_per_sec"] == (
        2 * 2 * block_size * 3 / final_metric["elapsed_sec"]
    )


def test_accumulated_gradients_match_one_effective_batch(tmp_path: Path) -> None:
    class Batches:
        def __init__(self, batches):
            self.batches = iter(batches)

        def get_batch(self, *args):
            return next(self.batches)

    config = GPTConfig(
        vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=0.0
    )
    torch.manual_seed(7)
    base = GPT(config)
    micro_batches = [
        (torch.randint(0, 16, (2, 4)), torch.randint(0, 16, (2, 4)))
        for _ in range(2)
    ]
    full_batch = tuple(torch.cat(parts) for parts in zip(*micro_batches))

    models = [GPT(config), GPT(config)]
    for model in models:
        model.load_state_dict(base.state_dict())
    trainers = [
        Trainer(
            models[0],
            config,
            TrainConfig(
                out_dir=str(tmp_path / "micro"), max_steps=1, warmup_steps=0,
                batch_size=2, grad_accum_steps=2, device="cpu",
            ),
            Batches(micro_batches),
            Batches([]),
        ),
        Trainer(
            models[1],
            config,
            TrainConfig(
                out_dir=str(tmp_path / "full"), max_steps=1, warmup_steps=0,
                batch_size=4, device="cpu",
            ),
            Batches([full_batch]),
            Batches([]),
        ),
    ]
    for trainer in trainers:
        trainer.estimate_loss = lambda: {"train": 1.0, "val": 1.0}
        trainer.train()

    for name, expected in models[1].state_dict().items():
        assert torch.allclose(models[0].state_dict()[name], expected, atol=1e-7), name


def test_early_stopping_saves_and_does_not_restart(tmp_path: Path) -> None:
    vocab_size = 16
    block_size = 4
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    make_synthetic_bin(train_path, vocab_size, 200, block_size)
    make_synthetic_bin(val_path, vocab_size, 100, block_size)
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
    )
    options = dict(
        out_dir=str(tmp_path / "early"),
        max_steps=10,
        warmup_steps=0,
        batch_size=2,
        eval_interval=2,
        eval_iters=1,
        early_stopping_patience=2,
        device="cpu",
    )
    trainer = Trainer(
        GPT(config),
        config,
        TrainConfig(**options),
        TokenDataset(train_path, block_size),
        TokenDataset(val_path, block_size),
    )
    losses = iter([1.0, 1.1, 1.2])
    trainer.estimate_loss = lambda: {"train": 1.0, "val": next(losses)}

    result = trainer.train()

    assert result["stopped_early"] is True
    assert result["completed_step"] == 4
    assert [metric["step"] for metric in result["metrics"]] == [0, 2, 4]
    last = torch.load(tmp_path / "early" / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(tmp_path / "early" / "best.pt", map_location="cpu", weights_only=False)
    assert (last["step"], last["no_improvement_evals"], last["stopped_early"]) == (4, 2, True)
    assert best["step"] == 0

    resumed = Trainer(
        GPT(config),
        config,
        TrainConfig(**options),
        TokenDataset(train_path, block_size),
        TokenDataset(val_path, block_size),
        resume_checkpoint=last,
    )
    resumed.optimizer.step = lambda: pytest.fail("stopped run performed an optimizer update")
    resumed.estimate_loss = lambda: pytest.fail("stopped run performed another evaluation")
    resumed_result = resumed.train()
    assert resumed_result["completed_step"] == 4
    assert resumed_result["stopped_early"] is True


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_gradient_accumulation_requires_a_positive_integer(value, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TrainConfig(out_dir=str(tmp_path), grad_accum_steps=value)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_early_stopping_requires_a_positive_integer(value, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TrainConfig(out_dir=str(tmp_path), early_stopping_patience=value)
