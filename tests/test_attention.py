import torch

from tinylm.model.attention import CausalSelfAttention
from tinylm.model.config import GPTConfig


def make_config(**overrides) -> GPTConfig:
    base = dict(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=8, dropout=0.0)
    base.update(overrides)
    return GPTConfig(**base)


def test_output_shape() -> None:
    config = make_config()
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 10, config.n_embd)
    out = attn(x)
    assert out.shape == (2, 10, config.n_embd)


def test_causal_mask_blocks_future_tokens() -> None:
    """Changing a future token must not change an earlier position's output."""
    torch.manual_seed(0)
    config = make_config()
    attn = CausalSelfAttention(config)
    attn.eval()

    x = torch.randn(1, 6, config.n_embd)
    x_modified = x.clone()
    x_modified[:, 3:, :] = torch.randn_like(x_modified[:, 3:, :])  # perturb positions 3..5

    with torch.no_grad():
        out_orig = attn(x)
        out_modified = attn(x_modified)

    # positions 0..2 only ever attend to positions <= themselves, so they
    # must be unaffected by changes to positions 3..5
    assert torch.allclose(out_orig[:, :3, :], out_modified[:, :3, :], atol=1e-6)
    # position 3 onward legitimately changed
    assert not torch.allclose(out_orig[:, 3:, :], out_modified[:, 3:, :], atol=1e-6)


def test_manual_and_fused_attention_agree() -> None:
    torch.manual_seed(0)
    config = make_config(dropout=0.0)

    manual = CausalSelfAttention(config, use_fused=False)
    fused = CausalSelfAttention(config, use_fused=True)
    fused.load_state_dict(manual.state_dict())
    manual.eval()
    fused.eval()

    x = torch.randn(2, 12, config.n_embd)
    with torch.no_grad():
        out_manual = manual(x)
        out_fused = fused(x)

    assert torch.allclose(out_manual, out_fused, atol=1e-4)


def test_rejects_incompatible_head_count() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_config(n_embd=10, n_head=3)
