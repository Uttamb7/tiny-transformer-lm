from __future__ import annotations

import json
from collections.abc import Sequence
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
    raw_text_path: Path | Sequence[Path],
    out_dir: Path,
    vocab_size: int = 512,
    val_fraction: float = 0.1,
    validation_text_path: Path | Sequence[Path] | None = None,
) -> dict:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [raw_text_path] if isinstance(raw_text_path, Path) else list(raw_text_path)
    if not paths:
        raise ValueError("at least one input file is required")
    documents = [path.read_text(encoding="utf-8") for path in paths]
    training_chars = sum(map(len, documents))
    if training_chars < 1000:
        raise ValueError(f"combined corpus looks too small ({training_chars} chars)")

    validation_paths = None
    validation_documents = None
    if validation_text_path is not None:
        validation_paths = (
            [validation_text_path]
            if isinstance(validation_text_path, Path)
            else list(validation_text_path)
        )
        if not validation_paths:
            raise ValueError("at least one validation input file is required")
        validation_documents = [path.read_text(encoding="utf-8") for path in validation_paths]
        if not sum(map(len, validation_documents)):
            raise ValueError("validation corpus is empty")

    tokenizer = ByteLevelBPETokenizer.train(documents, vocab_size=vocab_size, verbose=True)
    tokenizer.save(out_dir / "tokenizer.json")

    def encode(source_documents: list[str]) -> np.ndarray:
        ids = (
            tokenizer.encode(source_documents[0])
            if len(source_documents) == 1
            else tokenizer.encode_documents(source_documents)
        )
        return np.array(ids, dtype=np.uint16)

    train_ids = encode(documents)
    if validation_documents is None:
        split = int(len(train_ids) * (1 - val_fraction))
        train_ids, val_ids = train_ids[:split], train_ids[split:]
    else:
        val_ids = encode(validation_documents)
    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError("training and validation splits must each contain at least two tokens")

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    meta: dict[str, object] = {
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "source_chars": training_chars + sum(map(len, validation_documents or [])),
        "document_count": len(documents) + len(validation_documents or []),
        "compression_ratio_chars_per_token": round(
            (training_chars + sum(map(len, validation_documents or [])))
            / (len(train_ids) + len(val_ids)),
            3,
        ),
    }
    if validation_documents is not None:
        meta.update({
            "split": "documents",
            "train_document_count": len(documents),
            "val_document_count": len(validation_documents),
        })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
