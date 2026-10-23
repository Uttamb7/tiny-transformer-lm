from __future__ import annotations

import math
from collections.abc import Collection

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.model.attention import CausalSelfAttention, KVCache
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

    def forward_cached(
        self, x: torch.Tensor, cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        attended, cache = self.attn.forward_cached(self.ln_1(x), cache)
        x = x + attended
        x = x + self.mlp(self.ln_2(x))
        return x, cache


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

    def _forward_cached(
        self, idx: torch.Tensor, cache: list[KVCache] | None = None
    ) -> tuple[torch.Tensor, list[KVCache]]:
        past_length = cache[0][0].size(2) if cache else 0
        if past_length + idx.size(1) > self.config.block_size:
            raise ValueError("cached sequence exceeds block_size")
        pos = torch.arange(
            past_length, past_length + idx.size(1), dtype=torch.long, device=idx.device
        )
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        next_cache = []
        for layer, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            x, layer_cache = block.forward_cached(x, cache[layer] if cache else None)
            next_cache.append(layer_cache)
        return self.lm_head(self.ln_f(x)[:, [-1], :]), next_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
        suppress_token_ids: Collection[int] | None = None,
        use_cache: bool = True,
        generator: torch.Generator | None = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Append tokens, optionally stopping each row at a newly generated EOS.

        Finished rows retain EOS and are padded with it until every row finishes
        or the token budget is exhausted. EOS tokens in the prompt are ignored.
        """
        if idx.ndim != 2 or idx.size(0) < 1 or idx.size(1) < 1:
            raise ValueError("idx must be a non-empty 2D prompt tensor")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")
        if top_p is not None and (
            not isinstance(top_p, (int, float))
            or isinstance(top_p, bool)
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be finite and in (0, 1]")
        if (
            not isinstance(repetition_penalty, (int, float))
            or isinstance(repetition_penalty, bool)
            or not math.isfinite(repetition_penalty)
            or repetition_penalty < 1
        ):
            raise ValueError("repetition_penalty must be finite and at least 1")
        if no_repeat_ngram_size is not None and (
            not isinstance(no_repeat_ngram_size, int)
            or isinstance(no_repeat_ngram_size, bool)
            or no_repeat_ngram_size < 1
        ):
            raise ValueError("no_repeat_ngram_size must be a positive integer")
        if suppress_token_ids is None:
            suppressed: set[int] = set()
        elif isinstance(suppress_token_ids, (str, bytes)) or not isinstance(
            suppress_token_ids, Collection
        ):
            raise ValueError("suppress_token_ids must be a collection of vocabulary token IDs")
        else:
            suppressed = set(suppress_token_ids)
            if any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or not 0 <= token_id < self.config.vocab_size
                for token_id in suppressed
            ):
                raise ValueError("suppress_token_ids must contain only model vocabulary token IDs")
        if not isinstance(do_sample, bool):
            raise ValueError("do_sample must be a boolean")
        if eos_token_id is not None and (
            not isinstance(eos_token_id, int)
            or isinstance(eos_token_id, bool)
            or not 0 <= eos_token_id < self.config.vocab_size
        ):
            raise ValueError("eos_token_id must be an integer within the model vocabulary")
        if (
            not isinstance(min_new_tokens, int)
            or isinstance(min_new_tokens, bool)
            or min_new_tokens < 0
        ):
            raise ValueError("min_new_tokens must be a nonnegative integer")
        if min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens must not exceed max_new_tokens")
        if min_new_tokens and eos_token_id is None:
            raise ValueError("min_new_tokens requires eos_token_id")
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        cache: list[KVCache] | None = None
        for generated in range(max_new_tokens):
            if use_cache and cache is not None and cache[0][0].size(2) < self.config.block_size:
                logits, cache = self._forward_cached(idx[:, -1:], cache)
            else:
                idx_cond = idx[:, -self.config.block_size :]
                if use_cache:
                    logits, cache = self._forward_cached(idx_cond)
                else:
                    logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1:
                repeated = torch.gather(logits, 1, idx)
                repeated = torch.where(
                    repeated < 0,
                    repeated * repetition_penalty,
                    repeated / repetition_penalty,
                )
                logits.scatter_(1, idx, repeated)

            if suppressed:
                logits[:, list(suppressed)] = float("-inf")

            if no_repeat_ngram_size is not None:
                size = no_repeat_ngram_size
                for row in range(idx.size(0)):
                    if finished[row].item() or idx.size(1) + 1 < size:
                        continue
                    tokens = idx[row].tolist()
                    prefix = tokens[-(size - 1) :] if size > 1 else []
                    banned = {
                        tokens[start + size - 1]
                        for start in range(len(tokens) - size + 1)
                        if tokens[start : start + size - 1] == prefix
                    }
                    logits[row, list(banned)] = float("-inf")

            if eos_token_id is not None and generated < min_new_tokens:
                logits[:, eos_token_id] = float("-inf")

            exhausted = ~torch.isfinite(logits).any(dim=1) & ~finished
            if exhausted.any():
                raise ValueError("generation constraints banned every token")

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

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_id = next_id.masked_fill(finished[:, None], eos_token_id)
                finished |= next_id.squeeze(-1) == eos_token_id
            idx = torch.cat((idx, next_id), dim=1)
            if eos_token_id is not None and finished.all():
                break
        return idx
