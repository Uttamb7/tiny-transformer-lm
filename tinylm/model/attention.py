"""Causal multi-head self-attention, implemented explicitly rather than via
nn.MultiheadAttention or F.scaled_dot_product_attention, so the QK^T / mask /
softmax / weighted-sum mechanics are visible and testable.

`use_fused` switches to torch's fused scaled_dot_product_attention kernel for
a speed comparison against the manual implementation -- see benchmarks in the
README for the measured difference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.model.config import GPTConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, use_fused: bool = False) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_fused = use_fused

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout

        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.causal_mask: torch.Tensor
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project(x)
        y = self._attention(q, k, v, query_start=0)
        return self._output(y)

    def forward_cached(
        self, x: torch.Tensor, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        q, k, v = self._project(x)
        query_start = 0
        if cache is not None:
            query_start = cache[0].size(2)
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        y = self._attention(q, k, v, query_start)
        return self._output(y), (k, v)

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int
    ) -> torch.Tensor:
        if self.use_fused:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None
                if query_start == 0
                else self.causal_mask[
                    query_start : query_start + q.size(2), : k.size(2)
                ],
                is_causal=query_start == 0,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return self._manual_attention(q, k, v, query_start)

    def _output(self, y: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = y.shape
        C = self.n_embd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale

        mask = self.causal_mask[
            query_start : query_start + q.size(2), : k.size(2)
        ]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return attn_weights @ v  # (B, nh, T, hd)
