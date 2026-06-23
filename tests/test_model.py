"""Minimal smoke tests — run inside container with full dependency install."""
import sys
import torch
import torch.nn.functional as F

# ── Syntax checks ──────────────────────────────────────────────
def test_imports():
    from mini_model import MiniGPT, MiniGPTConfig
    from mini_tokenizer import BPETokenizer
    print("✓ Imports OK")

# ── Config + init ─────────────────────────────────────────────
def test_model_init():
    from mini_model import MiniGPT, MiniGPTConfig
    config = MiniGPTConfig(
        vocab_size=8192,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=256,
        eos_token_id=3,
    )
    model = MiniGPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    assert 7_000_000 < n_params < 8_000_000, f"Unexpected param count: {n_params:,}"
    print(f"✓ Model init OK ({n_params:,} params)")

# ── Forward pass ───────────────────────────────────────────────
def test_forward():
    from mini_model import MiniGPT, MiniGPTConfig
    config = MiniGPTConfig(vocab_size=8192, d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_seq_len=256, eos_token_id=3)
    model = MiniGPT(config)
    model.eval()
    x = torch.randint(0, 8192, (1, 16))
    with torch.no_grad():
        logits, _, _, _ = model(x, targets=None)
    assert logits.shape == (1, 16, 8192), f"Unexpected shape: {logits.shape}"
    print(f"✓ Forward pass OK — logits: {logits.shape}")

# ── Tokenizer ─────────────────────────────────────────────────
def test_tokenizer():
    from mini_tokenizer import BPETokenizer
    tok = BPETokenizer.load("/app/data/tokenizer.json")
    assert tok.vocab_size == 8192 or len(tok.token_to_id) == 8192
    ids = tok.encode("hello world")
    assert isinstance(ids, list) and len(ids) > 0
    print(f"✓ Tokenizer OK — 'hello world' → {len(ids)} tokens")

if __name__ == "__main__":
    tests = [
        test_imports,
        test_model_init,
        test_forward,
        test_tokenizer,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests PASSED")
