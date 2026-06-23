#!/usr/bin/env python3
"""
eval_quality.py — Benchmark mini-llm checkpoints.

Metrics:
  1. Perplexity on held-out test text (cross-entropy loss)
  2. Generation quality: token entropy at each position (lower = more confident)
  3. Repetition rate: how often the model repeats tokens within a generation

Usage:
  python eval_quality.py                    # eval all checkpoints
  python eval_quality.py --checkpoint sft_final.pt  # eval one checkpoint
"""

import argparse
import math
import os
import sys
import textwrap
import torch
import torch.nn.functional as F

from mini_model import MiniGPT, MiniGPTConfig
from mini_tokenizer import BPETokenizer


# ── Test prompts ──────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    "What is a transformer?",
    "How does attention work?",
    "Explain tokenization.",
    "What is KV caching?",
    "How does a neural network learn?",
    "What is a feedforward layer?",
    "Explain embeddings.",
    "What is a softmax?",
    "How does a language model generate text?",
    "What is fine-tuning?",
]


def load_model(checkpoint_path: str, tokenizer_path: str):
    tokenizer = BPETokenizer.load(tokenizer_path)
    config = MiniGPTConfig(
        vocab_size=8192,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=256,
        eos_token_id=tokenizer.eos_id,
    )
    model = MiniGPT(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model_keys = model.state_dict()
    filtered_state = {k: v for k, v in state.items() if k in model_keys and v.shape == model_keys[k].shape}
    import torch.nn as nn
    for k in sorted(set(model_keys) - set(filtered_state)):
        if k == "pos_embed.weight":
            old_pos = state[k]
            new_pos = nn.Embedding(256, config.d_model)
            new_pos.weight.data[:old_pos.shape[0]] = old_pos[:old_pos.shape[0]]
            model.pos_embed = new_pos
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    return model, tokenizer


def perplexity(model, tokenizer, text: str, block_size: int = 128) -> float:
    """
    Compute perplexity on a single text chunk.
    Uses sliding-window cross-entropy (shift-by-one targets).
    """
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) < 2:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0

    # Sliding windows — each position contributes one cross-entropy term
    for start in range(0, len(ids) - 1, block_size):
        chunk_ids = ids[start : start + block_size + 1]
        if len(chunk_ids) < 2:
            continue
        input_t = torch.tensor([chunk_ids[:-1]], dtype=torch.long)
        target_t = torch.tensor([chunk_ids[1:]], dtype=torch.long)

        with torch.no_grad():
            logits, _, _, _ = model.forward(input_t, targets=target_t)
            # logits: (1, T, vocab)  targets: (1, T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_t.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += target_t.numel()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def generation_metrics(model, tokenizer, prompt: str, max_new: int = 50) -> dict:
    """
    Run generation and collect quality metrics:
      - avg_logprob: mean log-prob of generated tokens (confidence)
      - repetition_ratio: unique_tokens / total_tokens (higher = less repetition)
      - unique_bigram_rate: unique_bigrams / total_bigrams
      - entropy: entropy of token distribution at each step, mean
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_t = torch.tensor([prompt_ids], dtype=torch.long)

    # Use model.generate to get output
    output_t = model.generate(
        input_t,
        max_new_tokens=max_new,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_id,
    )[0]

    gen_ids = output_t[len(prompt_ids) :].tolist()
    # Remove EOS
    if tokenizer.eos_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(tokenizer.eos_id)]

    if not gen_ids:
        return {"avg_logprob": float("nan"), "repetition_ratio": 0.0, "unique_bigram_rate": 0.0, "entropy": float("nan")}

    # Compute per-token logprobs via forward pass
    logprobs = []
    input_full = torch.tensor([prompt_ids + gen_ids], dtype=torch.long)
    with torch.no_grad():
        logits, _, _, _ = model.forward(input_full, targets=None)
        # logits: (1, seq_len, vocab)
        gen_logits = logits[0, len(prompt_ids) - 1 : -1, :]  # (gen_len, vocab)
        gen_tokens = torch.tensor(gen_ids, dtype=torch.long)
        log_probs = F.log_softmax(gen_logits, dim=-1)
        selected_lp = log_probs[range(len(gen_ids)), gen_tokens]  # (gen_len,)
        logprobs = selected_lp.tolist()

    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else float("nan")

    words = [tokenizer.id_to_token.get(t, "?") for t in gen_ids]
    words = [w.replace("</w>", "").replace("<bos>", "").replace("<eos>", "") for w in words]
    words = [w for w in words if w]

    total = len(words)
    unique = len(set(words))
    repetition_ratio = unique / total if total > 0 else 0.0

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    unique_bigrams = len(set(bigrams))
    unique_bigram_rate = unique_bigrams / len(bigrams) if bigrams else 0.0

    # Token distribution entropy
    import collections
    counter = collections.Counter(gen_ids)
    total_count = sum(counter.values())
    entropy = 0.0
    for count in counter.values():
        p = count / total_count
        entropy -= p * math.log2(p)

    decoded = tokenizer.decode(gen_ids)
    return {
        "avg_logprob": avg_logprob,
        "repetition_ratio": repetition_ratio,
        "unique_bigram_rate": unique_bigram_rate,
        "entropy": entropy,
        "decoded_preview": decoded[:120],
    }


def run_eval(checkpoint_path: str, tokenizer_path: str, test_text: str | None = None):
    """Full eval for one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}")

    model, tokenizer = load_model(checkpoint_path, tokenizer_path)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    # ── Perplexity ───────────────────────────────────────────────────
    if test_text:
        ppl = perplexity(model, tokenizer, test_text)
        print(f"\nPerplexity: {ppl:.2f}")
    else:
        print("\nPerplexity: (no test text provided)")
        ppl = None

    # ── Generation metrics ────────────────────────────────────────────
    print("\nGeneration Metrics:")
    print(f"{'Prompt':<45} {'AvgLogProb':>12} {'RepRatio':>8} {'UBR':>6}  {'Preview'}")
    print("-" * 110)

    for prompt in TEST_PROMPTS:
        metrics = generation_metrics(model, tokenizer, prompt, max_new=40)
        preview = metrics["decoded_preview"].replace("\n", " ")[:40]
        lp = f"{metrics['avg_logprob']:12.3f}" if not math.isnan(metrics["avg_logprob"]) else "         nan"
        rr = f"{metrics['repetition_ratio']:8.3f}"
        ubr = f"{metrics['unique_bigram_rate']:6.3f}"
        print(f"{prompt:<45} {lp} {rr} {ubr}  {preview}")

    return ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate mini-llm checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint to eval")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Path to checkpoints")
    parser.add_argument("--test-text", type=str, default=None, help="Path to test text file for perplexity")
    args = parser.parse_args()

    tokenizer_path = os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    # Load test text for perplexity
    test_text = None
    if args.test_text and os.path.exists(args.test_text):
        with open(args.test_text) as f:
            test_text = f.read()[:50_000]  # cap at 50k chars
        print(f"Loaded test text: {len(test_text):,} chars")

    if args.checkpoint:
        ckpts = [os.path.join(args.checkpoints_dir, args.checkpoint)]
    else:
        # Eval all SFT checkpoints
        ckpts = sorted(Path(args.checkpoints_dir).glob("sft*.pt"))

    print(f"\nEval config: {len(ckpts)} checkpoint(s)")
    print(f"Test prompts: {len(TEST_PROMPTS)}")

    results = {}
    for ckpt in ckpts:
        ppl = run_eval(str(ckpt), tokenizer_path, test_text)
        results[os.path.basename(ckpt)] = ppl

    # Summary
    print(f"\n{'='*60}")
    print("Summary — Perplexity")
    print(f"{'='*60}")
    for name, ppl in sorted(results.items(), key=lambda x: x[1] if x[1] else 999):
        if ppl is not None:
            print(f"  {name:<35} {ppl:.2f}")
        else:
            print(f"  {name:<35} {'N/A'}")


if __name__ == "__main__":
    main()
