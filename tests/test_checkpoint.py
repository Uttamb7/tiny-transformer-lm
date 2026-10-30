import json
from pathlib import Path

import pytest
import torch

from scripts.train import initialize_model
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
    model: GPT | None = None,
    **train_overrides,
) -> Trainer:
    import numpy as np

    for name, n in (("train.bin", 2000), ("val.bin", 500)):
        ids = np.random.default_rng(0).integers(0, config.vocab_size, size=n).astype(np.uint16)
        ids.tofile(tmp_path / name)

    train_data = TokenDataset(tmp_path / "train.bin", config.block_size)
    val_data = TokenDataset(tmp_path / "val.bin", config.block_size)
    train_options = {
        "out_dir": str(tmp_path / run_name),
        "device": "cpu",
        "batch_size": 8,
    }
    train_options.update(train_overrides)
    train_config = TrainConfig(**train_options)
    return Trainer(
        model if model is not None else GPT(config),
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
    assert list(tmp_path.glob(".ckpt.pt.*.tmp")) == []


def test_failed_checkpoint_save_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _make_trainer(tmp_path, make_config())
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous checkpoint")

    def fail_save(value, destination):
        Path(destination).write_bytes(b"partial checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        trainer.save_checkpoint(path, step=1, val_loss=2.0, best_val_loss=2.0)

    assert path.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.glob(".last.pt.*.tmp")) == []


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


def test_checkpoint_initialization_inherits_only_model_weights(tmp_path: Path) -> None:
    config = make_config()
    source = GPT(config)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    checkpoint = {
        "model_config": config.to_dict(),
        "model_state": source.state_dict(),
        "step": 999,
        "best_val_loss": 0.1,
        "optimizer_state": {"source": True},
    }
    target = GPT(config)

    initialize_model(target, config, checkpoint)
    trainer = _make_trainer(tmp_path, config, "initialized", model=target)

    for name, expected in source.state_dict().items():
        assert torch.equal(trainer.model.state_dict()[name], expected), name
    assert trainer.resume_step == 0
    assert trainer.best_val_loss == float("inf")
    assert trainer.optimizer.state == {}
    assert trainer.metrics == []
    assert trainer.resumed is False

    path = tmp_path / "initialized" / "last.pt"
    trainer.save_checkpoint(path, step=0, val_loss=2.0, best_val_loss=2.0)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _make_trainer(tmp_path, config, "initialized", resume_checkpoint=saved)


def test_checkpoint_initialization_validates_model_state() -> None:
    config = make_config()
    model = GPT(config)
    state = model.state_dict()

    with pytest.raises(ValueError, match="missing model_state"):
        initialize_model(model, config, {"model_config": config.to_dict()})
    with pytest.raises(ValueError, match="configuration does not match"):
        initialize_model(
            model, config,
            {"model_config": make_config(block_size=9).to_dict(), "model_state": state},
        )
    with pytest.raises(ValueError, match="model state is incompatible"):
        initialize_model(
            model, config,
            {
                "model_config": config.to_dict(),
                "model_state": {**state, "head.weight": torch.zeros(1)},
            },
        )


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

    legacy_default = {**checkpoint, "train_config": {**checkpoint["train_config"]}}
    del legacy_default["train_config"]["grad_accum_steps"]
    del legacy_default["train_config"]["early_stopping_patience"]
    del legacy_default["train_config"]["mixed_precision"]
    del legacy_default["train_config"]["mixed_precision_dtype"]
    del legacy_default["train_config"]["gradient_checkpointing"]
    del legacy_default["train_config"]["label_smoothing"]
    del legacy_default["grad_scaler_state"]
    _make_trainer(tmp_path, config, "legacy-default", resume_checkpoint=legacy_default)

    incomplete_mixed = {**checkpoint, "train_config": {**checkpoint["train_config"]}}
    incomplete_mixed["train_config"]["mixed_precision"] = True
    del incomplete_mixed["grad_scaler_state"]
    trainer.train_config.mixed_precision = True
    with pytest.raises(ValueError, match="missing grad_scaler_state"):
        trainer.restore_checkpoint(incomplete_mixed)

    with pytest.raises(ValueError, match="training configuration"):
        _make_trainer(
            tmp_path,
            config,
            "accum-mismatch",
            resume_checkpoint=checkpoint,
            grad_accum_steps=2,
        )

    with pytest.raises(ValueError, match="training configuration"):
        _make_trainer(
            tmp_path,
            config,
            "patience-mismatch",
            resume_checkpoint=checkpoint,
            early_stopping_patience=2,
        )

    dtype_mismatch = {**checkpoint, "train_config": {**checkpoint["train_config"]}}
    dtype_mismatch["train_config"]["mixed_precision_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="training configuration"):
        _make_trainer(
            tmp_path, config, "dtype-mismatch", resume_checkpoint=dtype_mismatch
        )

    checkpointing_mismatch = {**checkpoint, "train_config": {**checkpoint["train_config"]}}
    checkpointing_mismatch["train_config"]["gradient_checkpointing"] = True
    with pytest.raises(ValueError, match="training configuration"):
        _make_trainer(
            tmp_path, config, "checkpointing-mismatch",
            resume_checkpoint=checkpointing_mismatch,
        )

    smoothing_mismatch = {**checkpoint, "train_config": {**checkpoint["train_config"]}}
    smoothing_mismatch["train_config"]["label_smoothing"] = 0.1
    with pytest.raises(ValueError, match="training configuration"):
        _make_trainer(
            tmp_path, config, "smoothing-mismatch",
            resume_checkpoint=smoothing_mismatch,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mixed_precision_checkpoint_restores_scaler_after_training(tmp_path: Path) -> None:
    config = make_config()
    trainer = _make_trainer(
        tmp_path, config, "mixed", device="cuda", mixed_precision=True,
        max_steps=1, eval_interval=1, eval_iters=1, batch_size=2,
    )
    result = trainer.train()
    checkpoint = torch.load(
        tmp_path / "mixed" / "last.pt", map_location="cuda", weights_only=False
    )

    restored = _make_trainer(
        tmp_path, config, "mixed", device="cuda", mixed_precision=True,
        max_steps=1, eval_interval=1, eval_iters=1, batch_size=2,
        resume_checkpoint=checkpoint,
    )
    legacy_float16 = {**checkpoint, "train_config": {**checkpoint["train_config"]}}
    del legacy_float16["train_config"]["mixed_precision_dtype"]
    legacy_restored = _make_trainer(
        tmp_path, config, "mixed", device="cuda", mixed_precision=True,
        max_steps=1, eval_interval=1, eval_iters=1, batch_size=2,
        resume_checkpoint=legacy_float16,
    )

    assert result["completed_step"] == 1
    assert checkpoint["grad_scaler_state"] == trainer.scaler.state_dict()
    assert restored.scaler.state_dict() == trainer.scaler.state_dict()
    assert legacy_restored.scaler.state_dict() == trainer.scaler.state_dict()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA bfloat16 is required",
)
def test_bfloat16_checkpoint_resumes_without_gradient_scaling(tmp_path: Path) -> None:
    config = make_config()
    options = dict(
        device="cuda", mixed_precision=True, mixed_precision_dtype="bfloat16",
        max_steps=1, eval_interval=1, eval_iters=1, batch_size=2,
    )
    trainer = _make_trainer(tmp_path, config, "bfloat16", **options)
    result = trainer.train()
    checkpoint = torch.load(
        tmp_path / "bfloat16" / "last.pt", map_location="cuda", weights_only=False
    )
    restored = _make_trainer(
        tmp_path, config, "bfloat16", resume_checkpoint=checkpoint, **options
    )

    assert result["completed_step"] == 1
    assert trainer.scaler.is_enabled() is False
    assert checkpoint["train_config"]["mixed_precision_dtype"] == "bfloat16"
    assert restored.scaler.is_enabled() is False
