import pytest
import torch

import tinylm.model.transformer as transformer
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


def test_forward_label_smoothing_matches_cross_entropy() -> None:
    model = GPT(make_config())
    idx = torch.randint(0, model.config.vocab_size, (2, 4))
    targets = torch.randint(0, model.config.vocab_size, (2, 4))

    logits, loss = model(idx, targets, label_smoothing=0.2)

    expected = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.2
    )
    assert torch.equal(loss, expected)


def test_activation_checkpointing_only_wraps_gradient_enabled_training(monkeypatch) -> None:
    model = GPT(make_config(n_layer=2, dropout=0.0))
    model.gradient_checkpointing = True
    calls = 0
    real_checkpoint = transformer.checkpoint

    def counted_checkpoint(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(transformer, "checkpoint", counted_checkpoint)
    tokens = torch.randint(0, model.config.vocab_size, (2, 4))
    _, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    assert calls == model.config.n_layer

    model.eval()
    with torch.no_grad():
        model(tokens, tokens)
    model.generate(tokens[:, :1], max_new_tokens=1)
    assert calls == model.config.n_layer


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


@pytest.mark.parametrize(
    "prompt,kwargs,message",
    [
        (torch.empty(1, 0, dtype=torch.long), {}, "non-empty 2D"),
        (torch.ones(2, dtype=torch.long), {}, "non-empty 2D"),
        (torch.ones(1, 1, dtype=torch.long), {"max_new_tokens": -1}, "nonnegative"),
        (torch.ones(1, 1, dtype=torch.long), {"max_new_tokens": True}, "nonnegative"),
        (torch.ones(1, 1, dtype=torch.long), {"temperature": 0}, "finite and positive"),
        (torch.ones(1, 1, dtype=torch.long), {"temperature": float("inf")}, "finite and positive"),
        (torch.ones(1, 1, dtype=torch.long), {"top_k": 0}, "positive integer"),
        (torch.ones(1, 1, dtype=torch.long), {"top_p": 0}, "in \\(0, 1]"),
        (torch.ones(1, 1, dtype=torch.long), {"top_p": float("nan")}, "in \\(0, 1]"),
        (torch.ones(1, 1, dtype=torch.long), {"min_p": 0}, "in \\(0, 1]"),
        (torch.ones(1, 1, dtype=torch.long), {"min_p": float("inf")}, "in \\(0, 1]"),
        (torch.ones(1, 1, dtype=torch.long), {"repetition_penalty": 0.9}, "at least 1"),
        (torch.ones(1, 1, dtype=torch.long), {"repetition_penalty": float("inf")}, "at least 1"),
        (torch.ones(1, 1, dtype=torch.long), {"no_repeat_ngram_size": 0}, "positive integer"),
        (torch.ones(1, 1, dtype=torch.long), {"no_repeat_ngram_size": True}, "positive integer"),
        (torch.ones(1, 1, dtype=torch.long), {"suppress_token_ids": [-1]}, "vocabulary token IDs"),
        (torch.ones(1, 1, dtype=torch.long), {"suppress_token_ids": [64]}, "vocabulary token IDs"),
        (
            torch.ones(1, 1, dtype=torch.long),
            {"suppress_token_ids": [True]},
            "vocabulary token IDs",
        ),
        (torch.ones(1, 1, dtype=torch.long), {"suppress_token_ids": 1}, "collection"),
        (torch.ones(1, 1, dtype=torch.long), {"stop_sequences": [[]]}, "non-empty"),
        (torch.ones(1, 1, dtype=torch.long), {"stop_sequences": [[64]]}, "vocabulary"),
        (torch.ones(1, 1, dtype=torch.long), {"stop_sequences": [1]}, "token sequences"),
        (torch.ones(1, 1, dtype=torch.long), {"do_sample": "no"}, "boolean"),
    ],
)
def test_generate_rejects_invalid_settings_before_forward(
    monkeypatch, prompt, kwargs, message
) -> None:
    model = GPT(make_config())
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("invalid input reached model"))
    max_new_tokens = kwargs.get("max_new_tokens", 1)
    settings = {key: value for key, value in kwargs.items() if key != "max_new_tokens"}
    with pytest.raises(ValueError, match=message):
        model.generate(prompt, max_new_tokens=max_new_tokens, **settings)


def test_generate_zero_token_budget_is_a_noop() -> None:
    model = GPT(make_config())
    prompt = torch.tensor([[1, 2]])
    assert torch.equal(model.generate(prompt, max_new_tokens=0), prompt)


def test_greedy_generation_selects_each_rows_highest_eligible_logit(monkeypatch) -> None:
    model = GPT(make_config())

    def forward(idx):
        logits = torch.full((2, 1, model.config.vocab_size), -100.0)
        logits[0, 0, 4], logits[0, 0, 5] = 10.0, 9.0
        logits[1, 0, 6], logits[1, 0, 7] = 8.0, 7.0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: pytest.fail("sampled"))
    prompt = torch.tensor([[1], [2]])
    out = model.generate(
        prompt, max_new_tokens=1, do_sample=False, top_k=2, top_p=0.9, use_cache=False
    )
    assert out[:, -1].tolist() == [4, 6]


@pytest.mark.parametrize("repeated_logit,new_logit", [(10.0, 9.0), (-1.0, -1.5)])
def test_repetition_penalty_changes_each_rows_next_choice(
    monkeypatch, repeated_logit, new_logit
) -> None:
    model = GPT(make_config())

    def forward(idx):
        logits = torch.full((2, 1, model.config.vocab_size), -100.0)
        logits[0, 0, 1], logits[0, 0, 2] = repeated_logit, new_logit
        logits[1, 0, 2], logits[1, 0, 1] = repeated_logit, new_logit
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    prompt = torch.tensor([[1, 1], [2, 2]])
    default = model.generate(prompt, max_new_tokens=1, top_k=1, use_cache=False)
    penalized = model.generate(
        prompt, max_new_tokens=1, top_k=1, repetition_penalty=2, use_cache=False
    )
    assert default[:, -1].tolist() == [1, 2]
    assert penalized[:, -1].tolist() == [2, 1]


def test_min_p_filters_each_row_relative_to_its_best_token(monkeypatch) -> None:
    model = GPT(make_config())
    captured = []

    def forward(idx):
        logits = torch.full((2, 1, model.config.vocab_size), -100.0)
        logits[0, 0, 2], logits[0, 0, 3] = 10.0, 8.0
        logits[1, 0, 4], logits[1, 0, 5] = 5.0, 4.0
        return logits, None

    def sample(probs, num_samples, generator=None):
        captured.append(probs.clone())
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(torch, "multinomial", sample)
    model.generate(
        torch.tensor([[1], [1]]), max_new_tokens=1, min_p=0.2, use_cache=False
    )

    assert captured[0][0, 3] == 0
    assert captured[0][1, 5] > 0


def test_generate_is_deterministic_with_seed() -> None:
    config = make_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))

    torch.manual_seed(42)
    out1 = model.generate(idx.clone(), max_new_tokens=5, top_k=5)
    torch.manual_seed(42)
    out2 = model.generate(idx.clone(), max_new_tokens=5, top_k=5)

    assert torch.equal(out1, out2)


def test_generate_can_isolate_its_random_stream() -> None:
    torch.manual_seed(0)
    model = GPT(make_config())
    prompt = torch.tensor([[1, 2, 3]])
    global_state = torch.get_rng_state().clone()

    first = model.generate(
        prompt, max_new_tokens=8, top_k=10,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(torch.get_rng_state(), global_state)

    torch.rand(20)
    second = model.generate(
        prompt, max_new_tokens=8, top_k=10,
        generator=torch.Generator().manual_seed(42),
    )
    different = model.generate(
        prompt, max_new_tokens=8, top_k=10,
        generator=torch.Generator().manual_seed(43),
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_no_repeat_ngram_blocks_each_rows_repeated_phrase(monkeypatch) -> None:
    model = GPT(make_config())

    def forward(idx):
        logits = torch.full((2, 1, model.config.vocab_size), -100.0)
        logits[0, 0, 2], logits[0, 0, 3] = 10.0, 9.0
        logits[1, 0, 1], logits[1, 0, 3] = 10.0, 9.0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    prompt = torch.tensor([[1, 2, 1], [2, 1, 2]])
    default = model.generate(prompt, max_new_tokens=1, top_k=1, use_cache=False)
    constrained = model.generate(
        prompt, max_new_tokens=1, top_k=1, no_repeat_ngram_size=2, use_cache=False
    )
    assert default[:, -1].tolist() == [2, 1]
    assert constrained[:, -1].tolist() == [3, 3]


def test_no_repeat_ngram_reports_exhausted_vocabulary(monkeypatch) -> None:
    model = GPT(make_config(block_size=64))
    monkeypatch.setattr(
        model,
        "forward",
        lambda idx: (torch.zeros(1, 1, model.config.vocab_size), None),
    )
    prompt = torch.arange(model.config.vocab_size).unsqueeze(0)
    with pytest.raises(ValueError, match="banned every token"):
        model.generate(
            prompt, max_new_tokens=1, no_repeat_ngram_size=1, use_cache=False
        )


def test_generate_suppresses_tokens_and_reports_exhaustion(monkeypatch) -> None:
    model = GPT(make_config())

    def forward(idx):
        logits = torch.full((idx.size(0), 1, model.config.vocab_size), -100.0)
        logits[:, :, 2], logits[:, :, 3] = 10.0, 9.0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    prompt = torch.tensor([[1], [4]])
    default = model.generate(prompt, max_new_tokens=1, top_k=1, use_cache=False)
    suppressed = model.generate(
        prompt, max_new_tokens=1, top_k=1, suppress_token_ids=[2], use_cache=False
    )
    assert default[:, -1].tolist() == [2, 2]
    assert suppressed[:, -1].tolist() == [3, 3]

    with pytest.raises(ValueError, match="banned every token"):
        model.generate(
            prompt[:1], max_new_tokens=1,
            suppress_token_ids=range(model.config.vocab_size), use_cache=False,
        )


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
    out = model.generate(prompt, max_new_tokens=5, eos_token_id=eos_token_id, use_cache=False)
    assert out.tolist() == [[eos_token_id, 3, eos_token_id, eos_token_id]]
    assert len(calls) == 1


def test_generate_stops_batched_rows_on_complete_new_token_sequences(monkeypatch) -> None:
    model = GPT(make_config())
    choices = [[3, 2], [4, 3], [0, 4]]
    calls = []

    def forward(idx):
        logits = torch.full((2, 1, model.config.vocab_size), -100.0)
        for row, token in enumerate(choices[len(calls)]):
            logits[row, 0, token] = 100.0
        calls.append(idx.clone())
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    prompt = torch.tensor([[3, 4], [3, 4]])
    out = model.generate(
        prompt, max_new_tokens=5, stop_sequences=[[3, 4]], top_k=1, use_cache=False
    )

    assert out.tolist() == [[3, 4, 3, 4, 4], [3, 4, 2, 3, 4]]
    assert len(calls) == 3


def test_generate_delays_eos_until_minimum_new_tokens(monkeypatch) -> None:
    model = GPT(make_config())

    def forward(idx):
        logits = torch.full((idx.size(0), 1, model.config.vocab_size), -100.0)
        logits[:, :, 0] = 100.0
        logits[:, :, 2] = 50.0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    out = model.generate(
        torch.tensor([[1]]), max_new_tokens=5, eos_token_id=0,
        min_new_tokens=2, top_k=1, use_cache=False,
    )
    assert out.tolist() == [[1, 2, 2, 0]]


@pytest.mark.parametrize(
    "options,message",
    [
        ({"min_new_tokens": -1, "eos_token_id": 0}, "nonnegative"),
        ({"min_new_tokens": True, "eos_token_id": 0}, "nonnegative"),
        ({"min_new_tokens": 2, "max_new_tokens": 1, "eos_token_id": 0}, "must not exceed"),
        ({"min_new_tokens": 1}, "requires eos_token_id"),
    ],
)
def test_generate_rejects_invalid_minimum_before_forward(monkeypatch, options, message) -> None:
    model = GPT(make_config())

    def forward(idx):
        pytest.fail("invalid minimum reached the model")

    monkeypatch.setattr(model, "forward", forward)
    options = dict(options)
    max_new_tokens = options.pop("max_new_tokens", 2)
    with pytest.raises(ValueError, match=message):
        model.generate(torch.tensor([[1]]), max_new_tokens=max_new_tokens, **options)


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
    out = model.generate(
        prompt, max_new_tokens=budget, eos_token_id=0, top_k=1, top_p=0.9,
        use_cache=False,
    )
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
    out = model.generate(
        torch.tensor([[1]]), max_new_tokens=4, eos_token_id=eos_token_id, use_cache=False
    )
    assert out.tolist() == [[1, 2, 2, 2, 2]]


@pytest.mark.parametrize("eos_token_id", [-1, 64, 1.5, True])
def test_generate_rejects_invalid_eos_before_forward(monkeypatch, eos_token_id) -> None:
    model = GPT(make_config())

    def forward(idx):
        pytest.fail("invalid EOS reached the model")

    monkeypatch.setattr(model, "forward", forward)
    with pytest.raises(ValueError, match="eos_token_id"):
        model.generate(
            torch.tensor([[1]]), max_new_tokens=1, eos_token_id=eos_token_id,
            use_cache=False,
        )


@pytest.mark.parametrize("use_fused_attn", [False, True])
@pytest.mark.parametrize("do_sample", [False, True])
def test_cached_generation_matches_recomputation(use_fused_attn, do_sample) -> None:
    torch.manual_seed(0)
    model = GPT(make_config(block_size=6), use_fused_attn=use_fused_attn)
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])

    torch.manual_seed(42)
    cached = model.generate(
        prompt, max_new_tokens=7, top_k=1, repetition_penalty=1.2,
        min_p=0.05, no_repeat_ngram_size=2,
        suppress_token_ids=[0, 63], do_sample=do_sample,
    )
    torch.manual_seed(42)
    uncached = model.generate(
        prompt, max_new_tokens=7, top_k=1, repetition_penalty=1.2,
        min_p=0.05, no_repeat_ngram_size=2, suppress_token_ids=[0, 63],
        use_cache=False, do_sample=do_sample,
    )

    assert torch.equal(cached, uncached)


def test_cached_generation_reuses_tokens_until_the_window_slides(monkeypatch) -> None:
    model = GPT(make_config(block_size=5))
    original = model._forward_cached
    lengths = []

    def counted(idx, cache=None):
        lengths.append(idx.size(1))
        return original(idx, cache)

    monkeypatch.setattr(model, "_forward_cached", counted)
    model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4, top_k=1)

    assert lengths == [3, 1, 1, 5]


@pytest.mark.parametrize("use_fused_attn", [False, True])
def test_cached_generation_matches_batched_eos_stopping(use_fused_attn) -> None:
    torch.manual_seed(7)
    model = GPT(make_config(), use_fused_attn=use_fused_attn)
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.manual_seed(11)
    reference = model.generate(prompt, max_new_tokens=6, top_k=1, use_cache=False)
    eos_token_id = reference[0, prompt.size(1)].item()

    torch.manual_seed(11)
    cached = model.generate(
        prompt, max_new_tokens=6, top_k=1, eos_token_id=eos_token_id,
        no_repeat_ngram_size=2,
    )
    torch.manual_seed(11)
    uncached = model.generate(
        prompt, max_new_tokens=6, top_k=1, eos_token_id=eos_token_id,
        no_repeat_ngram_size=2, use_cache=False,
    )

    assert torch.equal(cached, uncached)
    assert cached[0, prompt.size(1) :].eq(eos_token_id).all()


@pytest.mark.parametrize("use_fused_attn", [False, True])
def test_cached_generation_matches_token_sequence_stopping(use_fused_attn) -> None:
    torch.manual_seed(7)
    model = GPT(make_config(), use_fused_attn=use_fused_attn)
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    reference = model.generate(prompt, max_new_tokens=2, top_k=1, use_cache=False)
    stop = reference[0, prompt.size(1) :].tolist()

    cached = model.generate(prompt, max_new_tokens=6, top_k=1, stop_sequences=[stop])
    uncached = model.generate(
        prompt, max_new_tokens=6, top_k=1, stop_sequences=[stop], use_cache=False
    )

    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("use_fused_attn", [False, True])
def test_minimum_generation_matches_cached_and_uncached(use_fused_attn) -> None:
    torch.manual_seed(7)
    model = GPT(make_config(), use_fused_attn=use_fused_attn)
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.manual_seed(11)
    reference = model.generate(prompt, max_new_tokens=1, top_k=1, use_cache=False)
    eos_token_id = reference[0, -1].item()

    torch.manual_seed(11)
    cached = model.generate(
        prompt, max_new_tokens=4, top_k=1, eos_token_id=eos_token_id,
        min_new_tokens=2,
    )
    torch.manual_seed(11)
    uncached = model.generate(
        prompt, max_new_tokens=4, top_k=1, eos_token_id=eos_token_id,
        min_new_tokens=2, use_cache=False,
    )

    assert torch.equal(cached, uncached)
    assert not cached[:, prompt.size(1) : prompt.size(1) + 2].eq(eos_token_id).any()
