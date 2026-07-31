import pytest
import torch

from tinylm.model import GPT, GPTConfig


def make_config(**overrides) -> GPTConfig:
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=8, dropout=0.0)
    base.update(overrides)
    return GPTConfig(**base)


def test_forward_shape_with_targets() -> None:
    config = make_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (3, 10))
    targets = torch.randint(0, config.vocab_size, (3, 10))

    logits, loss = model(idx, targets)
    assert logits.shape == (3, 10, config.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_forward_shape_without_targets_returns_last_position_only() -> None:
    config = make_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 10))

    logits, loss = model(idx)
    assert logits.shape == (2, 1, config.vocab_size)
    assert loss is None


def test_forward_rejects_sequence_longer_than_block_size() -> None:
    config = make_config(block_size=8)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 9))
    with pytest.raises(ValueError):
        model(idx)


def test_weight_tying() -> None:
    config = make_config()
    model = GPT(config)
    assert model.lm_head.weight is model.token_emb.weight


def test_generate_appends_requested_number_of_tokens() -> None:
    torch.manual_seed(0)
    config = make_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))

    out = model.generate(idx, max_new_tokens=7, temperature=1.0, top_k=10)
    assert out.shape == (1, 5 + 7)
    assert torch.equal(out[:, :5], idx)


def test_generate_is_deterministic_with_seed() -> None:
    config = make_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    torch.manual_seed(42)
    out1 = model.generate(idx.clone(), max_new_tokens=5, top_k=5)
    torch.manual_seed(42)
    out2 = model.generate(idx.clone(), max_new_tokens=5, top_k=5)

    assert torch.equal(out1, out2)


def test_num_params_excludes_positional_embedding_by_default() -> None:
    config = make_config()
    model = GPT(config)
    with_pos = model.num_params(non_embedding=False)
    without_pos = model.num_params(non_embedding=True)
    assert with_pos - without_pos == model.pos_emb.weight.numel()
