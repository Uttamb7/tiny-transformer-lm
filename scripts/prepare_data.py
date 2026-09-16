#!/usr/bin/env python
"""Download (if needed) a text corpus, train the BPE tokenizer on it, and
write tokenized train/val .bin files plus tokenizer.json + meta.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinylm.data.prepare import download_tinyshakespeare, prepare_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        nargs="+",
        default=None,
        help="One or more local text documents. If omitted, downloads TinyShakespeare.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data_raw" / "tinyshakespeare")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if args.input_file is None:
        raw_path = download_tinyshakespeare(REPO_ROOT / "data_raw" / "tinyshakespeare_raw.txt")
        print(f"downloaded corpus to {raw_path}")
    else:
        raw_path = args.input_file

    meta = prepare_dataset(
        raw_text_path=raw_path,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        val_fraction=args.val_fraction,
    )
    print(f"wrote dataset to {args.out_dir}")
    print(meta)


if __name__ == "__main__":
    main()
