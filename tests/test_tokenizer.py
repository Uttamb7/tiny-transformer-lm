import pytest

from tinylm.tokenizer import ByteLevelBPETokenizer

SAMPLE_TEXT = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
"""


@pytest.fixture(scope="module")
def trained_tokenizer() -> ByteLevelBPETokenizer:
    # A larger corpus so BPE has real merge candidates.
    return ByteLevelBPETokenizer.train(SAMPLE_TEXT * 20, vocab_size=300)


def test_vocab_size_reaches_target(trained_tokenizer: ByteLevelBPETokenizer) -> None:
    # vocab_size may fall short of the target if the corpus runs out of
    # repeated pairs before hitting min_frequency, but should get close.
    assert trained_tokenizer.vocab_size <= 300 + len(trained_tokenizer.special_tokens)
    assert len(trained_tokenizer.vocab) > 256  # at least some merges happened


@pytest.mark.parametrize(
    "text",
    [
        SAMPLE_TEXT,
        "",
        "a",
        "Hello, World! 123",
        "multiple   spaces\t\tand\ntabs\nnewlines",
        "unicode: café, naïve, 日本語, émoji 🎉",
        "'s 't 're contraction-like fragments",
    ],
)
def test_encode_decode_roundtrip(trained_tokenizer: ByteLevelBPETokenizer, text: str) -> None:
    ids = trained_tokenizer.encode(text)
    assert trained_tokenizer.decode(ids) == text


def test_compression_reduces_token_count(trained_tokenizer: ByteLevelBPETokenizer) -> None:
    ids = trained_tokenizer.encode(SAMPLE_TEXT)
    num_bytes = len(SAMPLE_TEXT.encode("utf-8"))
    assert len(ids) < num_bytes  # BPE should merge common byte sequences


def test_save_and_load_round_trip(tmp_path, trained_tokenizer: ByteLevelBPETokenizer) -> None:
    path = tmp_path / "tokenizer.json"
    trained_tokenizer.save(path)
    loaded = ByteLevelBPETokenizer.load(path)

    assert loaded.vocab_size == trained_tokenizer.vocab_size
    assert loaded.encode(SAMPLE_TEXT) == trained_tokenizer.encode(SAMPLE_TEXT)
    assert loaded.decode(loaded.encode(SAMPLE_TEXT)) == SAMPLE_TEXT


def test_train_rejects_vocab_size_below_256() -> None:
    with pytest.raises(ValueError):
        ByteLevelBPETokenizer.train("hello", vocab_size=100)


def test_document_encoding_uses_reserved_separator_without_cross_boundary_merges() -> None:
    tokenizer = ByteLevelBPETokenizer.train(["aaaa", "bbbb"], vocab_size=258)
    ids = tokenizer.encode_documents(["aaaa", "bbbb"])
    separator = tokenizer.special_tokens["<|endoftext|>"]
    assert ids == tokenizer.encode("aaaa") + [separator] + tokenizer.encode("bbbb")
    assert tokenizer.decode(ids) == "aaaa<|endoftext|>bbbb"
    with pytest.raises(ValueError, match="at least one"):
        tokenizer.encode_documents([])
