# train_sft.py
from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader

from mini_model import MiniGPT, MiniGPTConfig
from mini_tokenizer import BPETokenizer
from sft_dataset import SFTDataset


def train_sft():
    BATCH_SIZE = 8
    LR = 1e-4           # smaller LR for fine-tuning
    MAX_STEPS = 800     # SFT is fast
    BLOCK_SIZE = 128
    DEVICE = "cpu"

    # Load tokenizer
    tokenizer = BPETokenizer.load("data/tokenizer.json")

    # Load SFT dataset
    dataset = SFTDataset(tokenizer, "data/sft/instructions.jsonl", block_size=BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model config — must match pretrain config
    config = MiniGPTConfig(
        vocab_size=len(tokenizer.token_to_id),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=BLOCK_SIZE
    )

    model = MiniGPT(config).to(DEVICE)

    # Load pre-trained checkpoint
    state = torch.load("checkpoints/mini_gpt_pretrained.pt", map_location=DEVICE)
    model.load_state_dict(state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Starting SFT...")

    step = 0
    running_loss = 0.0

    model.train()

    while step < MAX_STEPS:
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"step {step}/{MAX_STEPS} | SFT loss: {avg_loss:.4f}")
                running_loss = 0.0

            if step >= MAX_STEPS:
                break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mini_gpt_sft.pt")
    print("SFT complete. Saved to checkpoints/mini_gpt_sft.pt")


if __name__ == "__main__":
    train_sft()
