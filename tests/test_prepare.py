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
    assert meta["min_frequency"] == 2


def test_prepare_dataset_configures_merge_frequency(tmp_path: Path) -> None:
    source = tmp_path / "pairs.txt"
    source.write_text("ab" * 600, encoding="utf-8")
    low_dir, high_dir = tmp_path / "low", tmp_path / "high"

    prepare_dataset(source, low_dir, vocab_size=257, min_frequency=2)
    high_meta = prepare_dataset(source, high_dir, vocab_size=257, min_frequency=1000)

    assert ByteLevelBPETokenizer.load(low_dir / "tokenizer.json").merges
    assert not ByteLevelBPETokenizer.load(high_dir / "tokenizer.json").merges
    assert high_meta["min_frequency"] == 1000


def test_prepare_cli_validates_merge_frequency(tmp_path: Path) -> None:
    source = tmp_path / "pairs.txt"
    source.write_text("ab" * 600, encoding="utf-8")
    out_dir = tmp_path / "configured"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"),
        "--input-file", str(source), "--out-dir", str(out_dir),
        "--vocab-size", "257", "--min-frequency", "1000",
    ]

    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert not ByteLevelBPETokenizer.load(out_dir / "tokenizer.json").merges
    assert json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))[
        "min_frequency"
    ] == 1000

    invalid = subprocess.run(
        command[:-1] + ["0"], capture_output=True, text=True, timeout=30
    )
    assert invalid.returncode == 2
    assert "--min-frequency must be a positive integer" in invalid.stderr


@pytest.mark.parametrize("min_frequency", [0, -1, 1.5, True])
def test_prepare_dataset_rejects_invalid_merge_frequency(
    tmp_path: Path, min_frequency
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        prepare_dataset([], tmp_path / "invalid", min_frequency=min_frequency)
    assert not (tmp_path / "invalid").exists()


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


def test_prepare_dataset_keeps_validation_documents_separate(tmp_path: Path) -> None:
    train_a, train_b = tmp_path / "train-a.txt", tmp_path / "train-b.txt"
    val_a, val_b = tmp_path / "val-a.txt", tmp_path / "val-b.txt"
    train_a.write_text("training alpha " * 50, encoding="utf-8")
    train_b.write_text("training beta " * 50, encoding="utf-8")
    val_a.write_text("z" * 40, encoding="utf-8")
    val_b.write_text("y" * 40, encoding="utf-8")
    out_dir = tmp_path / "explicit"

    meta = prepare_dataset(
        [train_a, train_b], out_dir, vocab_size=257,
        validation_text_path=[val_a, val_b],
    )
    tokenizer = ByteLevelBPETokenizer.load(out_dir / "tokenizer.json")
    train_ids = np.fromfile(out_dir / "train.bin", dtype=np.uint16).tolist()
    val_ids = np.fromfile(out_dir / "val.bin", dtype=np.uint16).tolist()

    assert (ord("z"), ord("z")) not in tokenizer.merges
    assert train_ids == tokenizer.encode_documents([
        train_a.read_text(encoding="utf-8"), train_b.read_text(encoding="utf-8")
    ])
    assert val_ids == tokenizer.encode_documents([
        val_a.read_text(encoding="utf-8"), val_b.read_text(encoding="utf-8")
    ])
    assert meta["split"] == "documents"
    assert meta["train_document_count"] == 2
    assert meta["val_document_count"] == 2
    assert meta["document_count"] == 4


def test_prepare_cli_accepts_dedicated_validation_documents(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text("train corpus " * 100, encoding="utf-8")
    validation.write_text("held out validation " * 10, encoding="utf-8")
    out_dir = tmp_path / "cli-explicit"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"),
        "--input-file", str(train),
        "--val-input-file", str(validation),
        "--out-dir", str(out_dir),
        "--vocab-size", "256",
    ]

    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    tokenizer = ByteLevelBPETokenizer.load(out_dir / "tokenizer.json")
    assert np.fromfile(out_dir / "train.bin", dtype=np.uint16).tolist() == tokenizer.encode(
        train.read_text(encoding="utf-8")
    )
    assert np.fromfile(out_dir / "val.bin", dtype=np.uint16).tolist() == tokenizer.encode(
        validation.read_text(encoding="utf-8")
    )

    missing_train = subprocess.run(
        [command[0], command[1], "--val-input-file", str(validation)],
        capture_output=True, text=True, timeout=30,
    )
    assert missing_train.returncode == 2
    assert "--val-input-file requires --input-file" in missing_train.stderr


def test_prepare_dataset_rejects_empty_validation_corpus(tmp_path: Path) -> None:
    train, validation = tmp_path / "train.txt", tmp_path / "validation.txt"
    train.write_text("training " * 150, encoding="utf-8")
    validation.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="validation corpus is empty"):
        prepare_dataset(train, tmp_path / "empty-val", validation_text_path=validation)


def test_prepare_dataset_rejects_missing_and_small_corpora(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one input"):
        prepare_dataset([], tmp_path / "missing")
    small = tmp_path / "small.txt"
    small.write_text("small", encoding="utf-8")
    with pytest.raises(ValueError, match="combined corpus"):
        prepare_dataset([small], tmp_path / "small-out")
