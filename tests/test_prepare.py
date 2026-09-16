import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tinylm.data.prepare import prepare_dataset
from tinylm.tokenizer import ByteLevelBPETokenizer


def written_ids(out_dir: Path) -> list[int]:
    return np.concatenate([
        np.fromfile(out_dir / "train.bin", dtype=np.uint16),
        np.fromfile(out_dir / "val.bin", dtype=np.uint16),
    ]).tolist()


def test_prepare_dataset_preserves_single_document_encoding(tmp_path: Path) -> None:
    source = tmp_path / "single.txt"
    source.write_text("single document " * 80, encoding="utf-8")
    out_dir = tmp_path / "single"
    meta = prepare_dataset(source, out_dir, vocab_size=256)
    tokenizer = ByteLevelBPETokenizer.load(out_dir / "tokenizer.json")
    assert written_ids(out_dir) == tokenizer.encode(source.read_text(encoding="utf-8"))
    assert meta["document_count"] == 1


def test_prepare_cli_inserts_real_document_boundaries(tmp_path: Path) -> None:
    first, second = tmp_path / "one.txt", tmp_path / "two.txt"
    first.write_text("a" * 600, encoding="utf-8")
    second.write_text("b" * 600, encoding="utf-8")
    out_dir = tmp_path / "multi"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"),
        "--input-file", str(first), str(second),
        "--out-dir", str(out_dir),
        "--vocab-size", "256",
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    tokenizer = ByteLevelBPETokenizer.load(out_dir / "tokenizer.json")
    ids = written_ids(out_dir)
    separator = tokenizer.special_tokens["<|endoftext|>"]
    assert ids == tokenizer.encode("a" * 600) + [separator] + tokenizer.encode("b" * 600)
    assert tokenizer.decode(ids) == "a" * 600 + "<|endoftext|>" + "b" * 600
    assert json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["document_count"] == 2


def test_prepare_dataset_rejects_missing_and_small_corpora(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one input"):
        prepare_dataset([], tmp_path / "missing")
    small = tmp_path / "small.txt"
    small.write_text("small", encoding="utf-8")
    with pytest.raises(ValueError, match="combined corpus"):
        prepare_dataset([small], tmp_path / "small-out")
