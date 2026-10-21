import subprocess
import sys
from pathlib import Path
from runpy import run_path

import torch

from tinylm.model import GPT, GPTConfig
from tinylm.tokenizer import ByteLevelBPETokenizer

generate_cli = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "generate.py")
)


def test_generate_cli_uses_dedicated_sampling_generator(tmp_path, monkeypatch) -> None:
    tokenizer = ByteLevelBPETokenizer.train("", vocab_size=256)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    captured = {}

    class Model:
        def generate(self, idx, **options):
            generator = options["generator"]
            captured["seed"] = generator.initial_seed()
            captured["draw"] = torch.rand((), generator=generator).item()
            return idx

    def load_model(checkpoint_path, device):
        torch.rand(10)  # Model loading must not consume the sampling stream.
        return Model(), {"step": 0, "val_loss": 0.0}

    monkeypatch.setitem(generate_cli["main"].__globals__, "load_model", load_model)
    monkeypatch.setattr(
        sys, "argv",
        ["generate.py", "--checkpoint", str(tmp_path / "unused.pt"),
         "--tokenizer", str(tokenizer_path), "--device", "cpu", "--seed", "42"],
    )
    generate_cli["main"]()

    assert captured["seed"] == 42
    assert captured["draw"] == torch.rand((), generator=torch.Generator().manual_seed(42)).item()


def test_generate_cli_requests_greedy_decoding(tmp_path, monkeypatch) -> None:
    tokenizer = ByteLevelBPETokenizer.train("", vocab_size=256)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    captured = {}

    class Model:
        def generate(self, idx, **options):
            captured.update(options)
            return idx

    monkeypatch.setitem(
        generate_cli["main"].__globals__,
        "load_model",
        lambda checkpoint_path, device: (Model(), {"step": 0, "val_loss": 0.0}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py", "--checkpoint", str(tmp_path / "unused.pt"),
            "--tokenizer", str(tokenizer_path), "--device", "cpu", "--greedy",
        ],
    )
    generate_cli["main"]()

    assert captured["do_sample"] is False


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
        model.token_emb.weight[ord("x")].fill_(0.5)
    checkpoint_path = tmp_path / "synthetic.pt"
    torch.save({"model_config": config.to_dict(), "model_state": model.state_dict(),
                "step": 0, "val_loss": 0.0}, checkpoint_path)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/generate.py"),
               "--checkpoint", str(checkpoint_path), "--tokenizer", str(tokenizer_path),
               "--device", "cpu", "--prompt", "Hi", "--max-new-tokens", "3", "--top-k", "1"]

    def run(*extra):
        return subprocess.run(command + list(extra), capture_output=True, text=True, timeout=30)

    stopped = run("--eos-token", "<|endoftext|>", "--no-repeat-ngram-size", "2")
    assert stopped.returncode == 0, stopped.stderr
    assert stopped.stdout.splitlines()[-1] == "Hi<|endoftext|>"
    default = run()
    assert default.returncode == 0, default.stderr
    assert default.stdout.splitlines()[-1] == "Hi" + "<|endoftext|>" * 3
    special_prompt = run(
        "--prompt", "<|endoftext|>", "--allowed-special", "<|endoftext|>",
        "--no-repeat-ngram-size", "1", "--max-new-tokens", "1",
    )
    assert special_prompt.returncode == 0, special_prompt.stderr
    assert special_prompt.stdout.splitlines()[-1] == "<|endoftext|>x"
    ordinary_prompt = run(
        "--prompt", "<|endoftext|>", "--no-repeat-ngram-size", "1",
        "--max-new-tokens", "1",
    )
    assert ordinary_prompt.returncode == 0, ordinary_prompt.stderr
    assert ordinary_prompt.stdout.splitlines()[-1] == "<|endoftext|><|endoftext|>"
    minimum = run("--eos-token", "<|endoftext|>", "--min-new-tokens", "2")
    assert minimum.returncode == 0, minimum.stderr
    assert minimum.stdout.splitlines()[-1] == "Hixx<|endoftext|>"
    batched = run("--num-samples", "3")
    assert batched.returncode == 0, batched.stderr
    assert batched.stdout.count("--- sample ") == 3
    assert batched.stdout.count("Hi" + "<|endoftext|>" * 3) == 3
    unknown = run("--eos-token", "<missing>")
    assert unknown.returncode == 2
    assert "unknown special token" in unknown.stderr
    invalid = run("--temperature", "0")
    assert invalid.returncode == 2
    assert "temperature must be finite and positive" in invalid.stderr
    invalid_penalty = run("--repetition-penalty", "0.5")
    assert invalid_penalty.returncode == 2
    assert "repetition_penalty must be finite and at least 1" in invalid_penalty.stderr
    invalid_samples = subprocess.run(
        [sys.executable, command[1], "--checkpoint", str(tmp_path / "missing.pt"),
         "--num-samples", "0"], capture_output=True, text=True, timeout=30,
    )
    assert invalid_samples.returncode == 2
    assert "must be a positive integer" in invalid_samples.stderr
    invalid_ngram = subprocess.run(
        [sys.executable, command[1], "--checkpoint", str(tmp_path / "missing.pt"),
         "--no-repeat-ngram-size", "0"], capture_output=True, text=True, timeout=30,
    )
    assert invalid_ngram.returncode == 2
    assert "must be a positive integer" in invalid_ngram.stderr
    unknown_prompt_token = subprocess.run(
        [sys.executable, command[1], "--checkpoint", str(tmp_path / "missing.pt"),
         "--tokenizer", str(tokenizer_path), "--allowed-special", "<missing>"],
        capture_output=True, text=True, timeout=30,
    )
    assert unknown_prompt_token.returncode == 2
    assert "unknown special token" in unknown_prompt_token.stderr
    missing_eos = run("--min-new-tokens", "1")
    assert missing_eos.returncode == 2
    assert "requires --eos-token" in missing_eos.stderr
    excessive_minimum = run(
        "--eos-token", "<|endoftext|>", "--max-new-tokens", "1",
        "--min-new-tokens", "2",
    )
    assert excessive_minimum.returncode == 2
    assert "must not exceed" in excessive_minimum.stderr
