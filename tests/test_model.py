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


@pytest.mark.parametrize("eos_token_id", [0, 63])
def test_generate_stops_at_first_new_eos(monkeypatch, eos_token_id) -> None:
    model = GPT(make_config())
    calls = []

    def forward(idx):
        calls.append(idx.clone())
        logits = torch.full((idx.size(0), 1, model.config.vocab_size), -100.0)
        logits[:, :, eos_token_id] = 100.0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    # An EOS already in the prompt must not stop a new completion.
    prompt = torch.tensor([[eos_token_id, 3, eos_token_id]])
    out = model.generate(prompt, max_new_tokens=5, eos_token_id=eos_token_id)
    assert out.tolist() == [[eos_token_id, 3, eos_token_id, eos_token_id]]
    assert len(calls) == 1


@pytest.mark.parametrize("budget", [0, 2, 5])
def test_generate_pads_finished_rows_and_respects_budget(monkeypatch, budget) -> None:
    model = GPT(make_config(block_size=2))
    # Row zero ends immediately; row one ends after three tokens.
    choices = [[0, 4], [7, 5], [8, 0]]
    calls = []

    def forward(idx):
        assert idx.size(1) <= model.config.block_size
        logits = torch.full((2, 1, model.config.vocab_size), -100.0)
        for row, token in enumerate(choices[len(calls)]):
            logits[row, 0, token] = 100.0
        calls.append(idx.clone())
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    prompt = torch.tensor([[1, 2], [2, 3]])
    out = model.generate(prompt, max_new_tokens=budget, eos_token_id=0, top_k=1, top_p=0.9)
    count = min(budget, 3)
    assert out.tolist() == [[1, 2] + [0] * count, [2, 3] + [4, 5, 0][:count]]
    assert len(calls) == count


@pytest.mark.parametrize("eos_token_id", [None, 63])
def test_generate_uses_full_budget_without_eos(monkeypatch, eos_token_id) -> None:
    model = GPT(make_config())

    def forward(idx):
        logits = torch.full((idx.size(0), 1, model.config.vocab_size), -100.0)
        logits[:, :, 2] = 100.0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    out = model.generate(torch.tensor([[1]]), max_new_tokens=4, eos_token_id=eos_token_id)
    assert out.tolist() == [[1, 2, 2, 2, 2]]


@pytest.mark.parametrize("eos_token_id", [-1, 64, 1.5, True])
def test_generate_rejects_invalid_eos_before_forward(monkeypatch, eos_token_id) -> None:
    model = GPT(make_config())

    def forward(idx):
        pytest.fail("invalid EOS reached the model")

    monkeypatch.setattr(model, "forward", forward)
    with pytest.raises(ValueError, match="eos_token_id"):
        model.generate(torch.tensor([[1]]), max_new_tokens=1, eos_token_id=eos_token_id)
