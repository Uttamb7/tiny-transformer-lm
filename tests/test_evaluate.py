import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tinylm.data.dataset import iter_eval_batches
from tinylm.model import GPT, GPTConfig


@pytest.mark.parametrize("length", [2, 4, 5, 9, 12, 25])
@pytest.mark.parametrize("batch_size", [1, 3, 20])
def test_eval_batches_cover_every_target_once(tmp_path, length, batch_size):
    path = tmp_path / "val.bin"
    np.arange(length, dtype=np.uint16).tofile(path)
    batches = list(iter_eval_batches(path, 4, batch_size, "cpu"))
    assert torch.cat([x.flatten() for x, _ in batches]).tolist() == list(range(length - 1))
    assert torch.cat([y.flatten() for _, y in batches]).tolist() == list(range(1, length))
    assert all(x.shape == y.shape and x.shape[0] <= batch_size and x.shape[1] <= 4
               for x, y in batches)


@pytest.mark.parametrize("contents,block,batch,message", [
    (b"", 4, 1, "at least two"), (b"\x00\x00", 4, 1, "at least two"),
    (b"\x00" * 5, 4, 1, "complete uint16"),
    (b"\x00" * 8, 0, 1, "positive"), (b"\x00" * 8, 4, 0, "positive"),
    (b"\x00" * 8, 4, -1, "positive"),
])
def test_eval_batches_reject_invalid_input(tmp_path, contents, block, batch, message):
    path = tmp_path / "val.bin"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match=message):
        list(iter_eval_batches(path, block, batch, "cpu"))


def test_eval_cli_matches_token_weighted_reference(tmp_path):
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=16, block_size=4, n_layer=1, n_head=1, n_embd=8,
                       dropout=0.5)
    model = GPT(config).eval()
    # Deliberately uneven per-token losses make tail weighting observable.
    with torch.no_grad():
        model.ln_f.bias.fill_(1)
        model.token_emb.weight[15].fill_(4)
    ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15], dtype=np.uint16)
    ids.tofile(tmp_path / "val.bin")
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"model_config": config.to_dict(), "model_state": model.state_dict()}, checkpoint)
    losses = []
    with torch.no_grad():
        for start in range(0, len(ids) - 1, config.block_size):
            x = torch.tensor(ids[start : start + config.block_size].astype(np.int64))[None]
            y = torch.tensor(ids[start + 1 : start + config.block_size + 1].astype(np.int64))
            x = x[:, :y.numel()]
            logits, _ = model(x, y[None])
            losses.extend(torch.nn.functional.cross_entropy(
                logits[0], y, reduction="none").tolist())
    expected = sum(losses) / len(losses)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/evaluate.py"),
               "eval", "--checkpoint", str(checkpoint), "--data-dir", str(tmp_path),
               "--device", "cpu"]
    for batch_size in [1, 3, 3]:
        run = subprocess.run(command + ["--batch-size", str(batch_size)],
                             capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout)
        assert result["num_tokens"] == len(ids) - 1
        assert result["num_batches"] == (3 if batch_size == 1 else 2)
        assert result["method"] == "non_overlapping_blocks"
        assert result["val_loss"] == pytest.approx(expected, rel=1e-6)
        assert result["val_perplexity"] == pytest.approx(math.exp(expected), rel=1e-5)
        assert result["val_perplexity"] > math.exp(20)  # no silent historical cap
    invalid = subprocess.run(command + ["--batch-size", "0"],
                             capture_output=True, text=True, timeout=30)
    assert invalid.returncode == 2
    assert "batch_size must be positive" in invalid.stderr
