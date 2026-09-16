"""Byte-level BPE tokenizer, implemented from scratch (no tiktoken / HF tokenizers).

Algorithm matches GPT-2's tokenizer: text is first split into chunks with a
regex so merges never cross word/whitespace boundaries, each chunk is turned
into its raw UTF-8 bytes, and the most frequent adjacent byte pair is merged
repeatedly until the target vocab size is reached. Operating on raw bytes
(0-255) as the base alphabet means every possible UTF-8 string round-trips
exactly, even if it contains characters never seen during training.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

# Approximation of GPT-2's pretokenization regex using ASCII character classes
# instead of \p{L}/\p{N} (which need the third-party `regex` module). Every
# character falls into exactly one alternative, so this still fully tiles any
# input string -- non-ASCII letters just land in the generic "other" bucket
# instead of the "letter" bucket, which only affects how efficiently they
# compress, not correctness.
PRETOKENIZE_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[A-Za-z]+"
    r"| ?[0-9]+"
    r"| ?[^\sA-Za-z0-9]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

NUM_BASE_BYTES = 256


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class ByteLevelBPETokenizer:
    def __init__(
        self,
        merges: dict[tuple[int, int], int],
        vocab: dict[int, bytes],
        special_tokens: dict[str, int] | None = None,
        pattern: str = PRETOKENIZE_PATTERN,
    ) -> None:
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens or {}
        self.special_id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @classmethod
    def train(
        cls,
        text: str | Iterable[str],
        vocab_size: int,
        special_tokens: tuple[str, ...] = ("<|endoftext|>",),
        pattern: str = PRETOKENIZE_PATTERN,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> ByteLevelBPETokenizer:
        if vocab_size < NUM_BASE_BYTES:
            raise ValueError(f"vocab_size must be >= {NUM_BASE_BYTES}, got {vocab_size}")

        compiled = re.compile(pattern)
        documents = (text,) if isinstance(text, str) else text
        chunk_freqs: Counter[str] = Counter()
        for document in documents:
            chunk_freqs.update(compiled.findall(document))

        word_to_ids: dict[str, list[int]] = {
            word: list(word.encode("utf-8")) for word in chunk_freqs
        }
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(NUM_BASE_BYTES)}
        merges: dict[tuple[int, int], int] = {}

        num_merges = vocab_size - NUM_BASE_BYTES
        for merge_step in range(num_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for word, ids in word_to_ids.items():
                freq = chunk_freqs[word]
                for a, b in zip(ids, ids[1:]):
                    pair_counts[(a, b)] += freq

            if not pair_counts:
                break

            best_pair, best_count = max(
                pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1])
            )
            if best_count < min_frequency:
                break

            new_id = NUM_BASE_BYTES + merge_step
            merges[best_pair] = new_id
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            for word in list(word_to_ids.keys()):
                ids = word_to_ids[word]
                if len(ids) >= 2:
                    word_to_ids[word] = _merge_pair(ids, best_pair, new_id)

            if verbose and merge_step % 100 == 0:
                merged_bytes = vocab[new_id]
                print(
                    f"merge {merge_step}/{num_merges}: {best_pair} -> {new_id} "
                    f"({merged_bytes!r}, count={best_count})"
                )

        next_id = NUM_BASE_BYTES + len(merges)
        special_ids = {}
        for tok in special_tokens:
            special_ids[tok] = next_id
            next_id += 1

        return cls(merges=merges, vocab=vocab, special_tokens=special_ids, pattern=pattern)

    def _encode_chunk(self, chunk_bytes: list[int]) -> list[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            candidate = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break
            ids = _merge_pair(ids, candidate, self.merges[candidate])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text with no special-token handling."""
        ids: list[int] = []
        for chunk in self._compiled.findall(text):
            chunk_bytes = list(chunk.encode("utf-8"))
            ids.extend(self._encode_chunk(chunk_bytes))
        return ids

    def encode(self, text: str) -> list[int]:
        return self.encode_ordinary(text)

    def encode_documents(
        self,
        documents: Iterable[str],
        separator_token: str = "<|endoftext|>",
    ) -> list[int]:
        documents = list(documents)
        if not documents:
            raise ValueError("at least one document is required")
        if separator_token not in self.special_tokens:
            raise ValueError(f"unknown special token: {separator_token!r}")
        separator_id = self.special_tokens[separator_token]
        ids: list[int] = []
        for index, document in enumerate(documents):
            if index:
                ids.append(separator_id)
            ids.extend(self.encode_ordinary(document))
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self.special_id_to_token:
                parts.append(self.special_id_to_token[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
            else:
                raise ValueError(f"Unknown token id: {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "pattern": self.pattern,
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ByteLevelBPETokenizer:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(NUM_BASE_BYTES)}
        merges: dict[tuple[int, int], int] = {}
        for a, b, new_id in data["merges"]:
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]

        special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        return cls(
            merges=merges,
            vocab=vocab,
            special_tokens=special_tokens,
            pattern=data["pattern"],
        )
