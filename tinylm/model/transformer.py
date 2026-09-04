from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.model.attention import CausalSelfAttention
from tinylm.model.config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, use_fused_attn: bool = False) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, use_fused=use_fused_attn)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, use_fused_attn: bool = False) -> None:
        super().__init__()
        self.config = config
        self.use_fused_attn = use_fused_attn

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [Block(config, use_fused_attn=use_fused_attn) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: input embedding and output projection share weights
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Append tokens, optionally stopping each row at a newly generated EOS.

        Finished rows retain EOS and are padded with it until every row finishes
        or the token budget is exhausted. EOS tokens in the prompt are ignored.
        """
        if eos_token_id is not None and (
            not isinstance(eos_token_id, int)
            or isinstance(eos_token_id, bool)
            or not 0 <= eos_token_id < self.config.vocab_size
        ):
            raise ValueError("eos_token_id must be an integer within the model vocabulary")
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            if idx.size(1) <= self.config.block_size:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                sorted_mask = cum_probs - probs > top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                next_id = next_id.masked_fill(finished[:, None], eos_token_id)
                finished |= next_id.squeeze(-1) == eos_token_id
            idx = torch.cat((idx, next_id), dim=1)
            if eos_token_id is not None and finished.all():
                break
        return idx
