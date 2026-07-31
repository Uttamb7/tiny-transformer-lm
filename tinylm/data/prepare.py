from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from tinylm.tokenizer import ByteLevelBPETokenizer

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tinyshakespeare(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urlopen(TINYSHAKESPEARE_URL, timeout=30) as resp:  # noqa: S310 - fixed, known URL
            text = resp.read().decode("utf-8")
        dest.write_text(text, encoding="utf-8")
    return dest


def prepare_dataset(
    raw_text_path: Path,
    out_dir: Path,
    vocab_size: int = 512,
    val_fraction: float = 0.1,
) -> dict:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    text = raw_text_path.read_text(encoding="utf-8")
    if len(text) < 1000:
        raise ValueError(f"corpus at {raw_text_path} looks too small ({len(text)} chars)")

    tokenizer = ByteLevelBPETokenizer.train(text, vocab_size=vocab_size, verbose=True)
    tokenizer.save(out_dir / "tokenizer.json")

    ids = tokenizer.encode(text)
    ids_arr = np.array(ids, dtype=np.uint16)

    n = len(ids_arr)
    split = int(n * (1 - val_fraction))
    train_ids, val_ids = ids_arr[:split], ids_arr[split:]

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "source_chars": len(text),
        "compression_ratio_chars_per_token": round(len(text) / n, 3),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
