# generate_text.py
from __future__ import annotations

import argparse
import torch

from mini_model import MiniGPT, MiniGPTConfig
from mini_tokenizer import BPETokenizer


def load_model(checkpoint_path: str, block_size: int = 128):    # Load tokenizer to get vocab size
    tokenizer = BPETokenizer.load("data/tokenizer.json")

    config = MiniGPTConfig(
        vocab_size=len(tokenizer.token_to_id),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=block_size,
    )

    model = MiniGPT(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, tokenizer


def generate_text(
    prompt: str,
    checkpoint_path: str = "checkpoints/mini_gpt_pretrained.pt",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
):
    model, tokenizer = load_model(checkpoint_path)
    device = "cpu"
    model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    out_ids = out_ids[0].tolist()
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mini_gpt_pretrained.pt",
    )
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.0)

    args = parser.parse_args()

    top_p = args.top_p if args.top_p > 0 else None
    text = generate_text(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=top_p,
    )

    print("=== GENERATED TEXT ===")
    print(text)


if __name__ == "__main__":
    main()
