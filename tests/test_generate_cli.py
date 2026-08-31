import subprocess
import sys
from pathlib import Path

import torch

from tinylm.model import GPT, GPTConfig
from tinylm.tokenizer import ByteLevelBPETokenizer


def test_generate_cli_optional_eos_and_unknown_token(tmp_path: Path) -> None:
    tokenizer = ByteLevelBPETokenizer.train("", vocab_size=256)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    eos = tokenizer.special_tokens["<|endoftext|>"]
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=8,
                       n_layer=1, n_head=1, n_embd=4, dropout=0.0)
    model = GPT(config)
    # A synthetic checkpoint that deterministically prefers EOS; not a trained model.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ln_f.bias.fill_(1)
        model.token_emb.weight[eos].fill_(1)
    checkpoint_path = tmp_path / "synthetic.pt"
    torch.save({"model_config": config.to_dict(), "model_state": model.state_dict(),
                "step": 0, "val_loss": 0.0}, checkpoint_path)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/generate.py"),
               "--checkpoint", str(checkpoint_path), "--tokenizer", str(tokenizer_path),
               "--device", "cpu", "--prompt", "Hi", "--max-new-tokens", "3", "--top-k", "1"]

    def run(*extra):
        return subprocess.run(command + list(extra), capture_output=True, text=True, timeout=30)

    stopped = run("--eos-token", "<|endoftext|>")
    assert stopped.returncode == 0, stopped.stderr
    assert stopped.stdout.splitlines()[-1] == "Hi<|endoftext|>"
    default = run()
    assert default.returncode == 0, default.stderr
    assert default.stdout.splitlines()[-1] == "Hi" + "<|endoftext|>" * 3
    unknown = run("--eos-token", "<missing>")
    assert unknown.returncode == 2
    assert "unknown special token" in unknown.stderr
