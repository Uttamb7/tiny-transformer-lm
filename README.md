# Tiny Transformer Language Model

A GPT-style transformer language model built from scratch: a byte-level BPE tokenizer, causal multi-head self-attention, and a training/evaluation loop, trained on the TinyShakespeare corpus.

## Why this project exists

Most of my other projects are full-stack/backend work (React, Next.js, FastAPI, Postgres). This project exists to demonstrate the machine-learning engineering that isn't visible in that work: implementing attention mechanics directly, training a tokenizer from raw text, running real training/validation loops, and measuring model quality instead of just calling a hosted LLM API.

## Main features

- Byte-level BPE tokenizer trained from scratch (no tiktoken/HuggingFace tokenizers) — always round-trips arbitrary UTF-8 exactly, even on unseen characters
- Causal multi-head self-attention implemented manually (explicit QK^T / mask / softmax), with an optional fused `scaled_dot_product_attention` path for a measured speed comparison
- Configurable model sizes (nano / tiny / small) via JSON configs
- Training loop with AdamW, linear warmup + cosine LR decay, gradient clipping, NaN/divergence detection, and periodic checkpointing (best + last)
- Text generation with key/value-cached decoding, temperature / top-k / top-p sampling, and optional end-token stopping
- Evaluation tooling: exact validation perplexity, cross-config comparison reports, manual-vs-fused attention benchmarks

## Technology stack

Python 3.12, PyTorch 2.6 (CUDA 12.4), NumPy. Testing via pytest + pytest-cov. Linting/type-checking via ruff + mypy.

## Architecture summary

```
tinylm/
  tokenizer/   byte-level BPE: train, encode, decode, save/load
  model/       GPTConfig, CausalSelfAttention, Block/MLP, GPT (forward + generate)
  data/        TokenDataset (memmapped .bin token files, random-window batching)
               prepare.py (corpus -> tokenizer + train.bin/val.bin)
  training/    Trainer (train/eval loop, checkpointing), LR schedule
scripts/       CLI entry points: prepare_data.py, train.py, generate.py, evaluate.py
configs/       nano.json / tiny.json / small.json model+training hyperparameters
tests/         unit tests (tokenizer, attention, model, LR schedule, checkpoints)
               + integration smoke test (real training steps on synthetic data)
```

Data flow: raw text -> BPE tokenizer training -> `train.bin`/`val.bin` (uint16 token ids) -> `TokenDataset` samples random contiguous windows -> `GPT` forward pass -> cross-entropy loss -> AdamW update. Checkpoints store the exact `GPTConfig` they were trained with, so loading a checkpoint never depends on an external config file staying in sync.

This is a CLI/library project, not a web service — no API layer, since the point is the model internals, not another REST wrapper.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # or `source .venv/bin/activate` on Linux/macOS
pip install -e . -r requirements.txt
```

Requires a working PyTorch install; GPU (CUDA) is optional but training is much faster with one.

## Run

```bash
# 1. Download TinyShakespeare, train the BPE tokenizer, write train/val .bin files
python scripts/prepare_data.py --vocab-size 512

# 2. Train a model (nano / tiny / small)
python scripts/train.py --config tiny

# Resume an interrupted run from its latest checkpoint
python scripts/train.py --config tiny --resume runs/tiny/last.pt

# Use four micro-batches per optimizer step when memory is constrained
python scripts/train.py --config tiny --grad-accum-steps 4

# Stop after four validation checks without a new best loss
python scripts/train.py --config tiny --early-stopping-patience 4

# 3. Generate text from a checkpoint
python scripts/generate.py --checkpoint runs/tiny/best.pt --prompt "ROMEO:" --max-new-tokens 200

# Optional: stop when a tokenizer special token is generated
python scripts/generate.py --checkpoint runs/tiny/best.pt --prompt "ROMEO:" --eos-token "<|endoftext|>"

# 4. Evaluate / compare
python scripts/evaluate.py eval --checkpoint runs/tiny/best.pt
python scripts/evaluate.py compare runs/nano runs/tiny runs/small
python scripts/evaluate.py benchmark-attn --device cuda
```

Generation uses the full token budget by default. `--eos-token` must name a special
token in the loaded tokenizer; the generated end token is retained in the output.
The library equivalent is `GPT.generate(..., eos_token_id=token_id)`: each batch
row stops at its first newly generated end token and is padded with that token
until all rows finish or the budget runs out. End tokens already in the prompt
do not stop a new completion. Existing TinyShakespeare training does not insert
document-boundary tokens, so stopping does not teach those checkpoints when to end.
Generation reuses each layer's attention keys and values while the active context
fits within `block_size`. When the context window slides, it recomputes the cropped
window so learned positional embeddings keep exactly the same behavior as uncached
generation. Pass `use_cache=False` to `GPT.generate` for a reference recomputation.

Training checkpoints include the optimizer, completed step, best validation loss,
CPU/trainer random states, device type, attention implementation, and
schedule-critical training configuration. Resume uses the checkpoint's directory
and appends to its existing `metrics.jsonl`; that log must end at the checkpoint
step. Use the same config, device type, and `--use-fused-attn` setting. Older
checkpoints without resume metadata fail clearly and remain usable for generation
and evaluation.

`--grad-accum-steps` averages that many independently sampled micro-batches before
one clipped optimizer update. Learning-rate, evaluation, and checkpoint steps remain
optimizer-step based, while throughput includes every accumulated token. Resume
requires the same setting; checkpoints created before this option use the default of 1.

`--early-stopping-patience` stops after that many consecutive scheduled validation
checks fail to improve validation loss. The stopping check is retained in
`metrics.jsonl` and `last.pt`; `best.pt` remains the best validation checkpoint.
The default is disabled. Resume requires the same patience and preserves its count;
resuming an already stopped run performs no additional optimizer update. Older
checkpoints use the disabled default.

`evaluate.py eval` scores every next-token target exactly once in deterministic,
non-overlapping blocks, resetting context and position indices at each block.
It includes the final partial block and weights loss by target count; JSON output
reports `num_tokens` (file tokens minus one), `num_batches`, `block_size`, and
`method`. `--batch-size` controls memory usage without changing target coverage.
This is full-set evaluation with block-local context, not sliding-window perplexity.
Validation files need at least two uint16 tokens. Training still uses sampled windows.

## Test

```bash
ruff check .
mypy tinylm scripts
pytest -q --cov=tinylm --cov-report=term-missing
```

Tests cover: BPE encode/decode round-trip (including empty strings and non-ASCII text), causal-mask correctness (perturbing future tokens must not change earlier outputs), manual-vs-fused attention numerical agreement, model forward-shape and weight-tying checks, optional end-token stopping (including batched padding and CLI behavior), LR schedule math, checkpoint save/load equivalence, an integration smoke test that trains a tiny model on a synthetic repeating pattern and asserts loss actually drops, and a divergence test that asserts a runaway learning rate raises a clear error instead of silently producing NaNs.

## Verified results

All numbers below were produced by actually running the commands above on this machine (RTX 4070 SUPER, CUDA 12.4) against the TinyShakespeare corpus (1,115,394 characters, tokenized to 575,345 tokens at a measured 1.939 chars/token compression ratio with a 512-entry BPE vocab).

| config | params (non-embed) | val loss | val perplexity | train time | device |
|--------|--------------------:|---------:|---------------:|-----------:|--------|
| nano   | 132,928             | 3.614    | 37.12           | 12.4s      | cuda   |
| tiny   | 859,008             | 2.887    | 17.93           | 134.8s     | cuda   |
| small  | 10,844,544          | 2.843    | 17.16           | 1036.0s    | cuda   |

(Historical val loss/perplexity above came from the previous `scripts/evaluate.py eval`,
which sampled random windows rather than covering the full validation set. These
numbers have not been recomputed with the deterministic evaluator and should not
be compared directly with its results.)

**Diminishing returns**: small has 12.6x the parameters of tiny and trains 7.7x longer, but only improves validation perplexity from 17.93 to 17.16 (4.3%). Its training curve (`runs/small/metrics.jsonl`) also shows the best checkpoint landing at step 1500/5000 — validation loss bottoms out at 2.833 and then rises again while training loss keeps falling, i.e. the model overfits the ~518K-token training set well before `max_steps` is reached. The `Trainer` saves `best.pt` on every validation improvement specifically so this doesn't require early-stopping logic to still get the best checkpoint. On a corpus this small, model capacity beyond "tiny" buys very little, and the config comparison is what surfaces that instead of assuming bigger is better.

Attention benchmark (batch=32, block=256, n_head=6, n_embd=384, 100 iters, RTX 4070 SUPER):

| implementation | iters/sec | relative |
|---|---:|---:|
| manual (explicit QK^T + softmax) | 168.4 | 1.0x |
| fused `scaled_dot_product_attention` | 387.6 | 2.30x |

Sample generation (tiny config, prompt `"ROMEO:"`, temperature 0.5, top-k 50, seed 42):

```
ROMEO:
Ay, art thou, very let'st thousand and true.

ROMEO:
It is a word, and I will never be than it in then,
And let me see him took.

KING RICHARD II:
Reland, madam, it is in a sacred words and
And Bolingbroke his son, and not the pardon
It is a plainted and Henry Saint Henry,
That I may see his s
```

Test coverage: 88% line coverage on `tinylm/` (`pytest --cov`); the untested module is `data/prepare.py`, which is thin glue around the (separately unit-tested) tokenizer training/save calls plus a network download, and is instead exercised directly via `scripts/prepare_data.py`.

## Current limitations

- Generated text is not grammatically coherent — these are intentionally small models (132K-2M non-embedding params) trained briefly on ~500K training tokens, not production-scale LLMs. The goal was demonstrating the training/eval mechanics, not chasing fluency.
- The pretokenization regex approximates GPT-2's pattern using ASCII character classes instead of Unicode `\p{L}`/`\p{N}` (to avoid the third-party `regex` dependency). Non-ASCII text still round-trips exactly through the byte-level fallback, it just compresses less efficiently.
- No distributed/multi-GPU training — single-device only, appropriate for this scale.
- `<|endoftext|>` is reserved in the vocabulary but not currently inserted between documents, since TinyShakespeare is treated as one continuous corpus.
