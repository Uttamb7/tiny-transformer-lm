from pathlib import Path

import numpy as np
import pytest
import torch

from tinylm.data.dataset import TokenDataset


def write_bin(path: Path, ids: list[int]) -> None:
    np.array(ids, dtype=np.uint16).tofile(path)


def test_get_batch_shapes_and_offset_by_one(tmp_path: Path) -> None:
    path = tmp_path / "train.bin"
    write_bin(path, list(range(1000)))
    ds = TokenDataset(path, block_size=8)

    x, y = ds.get_batch(batch_size=4, device="cpu", generator=torch.Generator().manual_seed(0))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted by one token, since token ids here are just their own index
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_rejects_dataset_smaller_than_block_size(tmp_path: Path) -> None:
    path = tmp_path / "tiny.bin"
    write_bin(path, list(range(4)))
    with pytest.raises(ValueError):
        TokenDataset(path, block_size=8)
