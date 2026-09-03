"""Binary token-id storage and random-window batch sampling for LM training.

Token ids are memory-mapped from disk (np.uint16) rather than loaded fully
into RAM, and re-mapped fresh on every batch call, which avoids a known
PyTorch DataLoader + np.memmap leak where the mapped file handle is kept
alive and slowly grows the process's memory footprint over long training runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch


def iter_eval_batches(
    bin_path: str | Path, block_size: int, batch_size: int, device: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Score every next-token target once, resetting context at each block.

    Full blocks are batched; a shorter final block is yielded separately so
    no target is dropped or padded. Only the current batch is copied into RAM.
    """
    if block_size < 1 or batch_size < 1:
        raise ValueError("block_size and batch_size must be positive")
    path = Path(bin_path)
    size = path.stat().st_size
    if size % 2:
        raise ValueError("validation file must contain complete uint16 tokens")
    if size < 4:
        raise ValueError("validation file must contain at least two tokens")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    targets = len(data) - 1
    full_end = targets - targets % block_size
    for start in range(0, full_end, batch_size * block_size):
        end = min(start + batch_size * block_size, full_end)
        x = torch.from_numpy(data[start:end].astype(np.int64)).reshape(-1, block_size)
        y = torch.from_numpy(data[start + 1 : end + 1].astype(np.int64)).reshape(-1, block_size)
        yield x.to(device), y.to(device)
    if full_end < targets:
        x = torch.from_numpy(data[full_end:-1].astype(np.int64)).unsqueeze(0)
        y = torch.from_numpy(data[full_end + 1 :].astype(np.int64)).unsqueeze(0)
        yield x.to(device), y.to(device)


class TokenDataset:
    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        self.bin_path = Path(bin_path)
        self.block_size = block_size
        self._length = len(np.memmap(self.bin_path, dtype=np.uint16, mode="r"))
        if self._length <= block_size:
            raise ValueError(
                f"dataset {self.bin_path} has {self._length} tokens, "
                f"which is not enough for block_size={block_size}"
            )

    def __len__(self) -> int:
        return self._length - self.block_size

    def get_batch(
        self, batch_size: int, device: str, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        max_start = len(data) - self.block_size - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)

        x = torch.stack(
            [torch.from_numpy(data[s : s + self.block_size].astype(np.int64)) for s in starts]
        )
        y = torch.stack(
            [
                torch.from_numpy(data[s + 1 : s + 1 + self.block_size].astype(np.int64))
                for s in starts
            ]
        )

        if device.startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
