# train_pretrain.py
from __future__ import annotations

import os
import time
import torch
from torch.utils.data import DataLoader

from mini_model import MiniGPT, MiniGPTConfig
from mini_tokenizer import BPETokenizer
from dataset import PretrainDataset


def train():
    # ----------------------------------------------------
    # Config
    # ----------------------------------------------------
    BLOCK_SIZE = 128
    BATCH_SIZE = 16
    LR = 3e-4
    MAX_STEPS = 2000
    DEVICE = "cpu"   # keep it CPU-friendly

    # ----------------------------------------------------
    # Load tokenizer + text
    # ----------------------------------------------------
    tokenizer = BPETokenizer.load("data/tokenizer.json")
    raw_text = open("data/raw/pretrain.txt", "r", encoding="utf-8").read()

    # Tokenize entire file
    token_ids = tokenizer.encode(raw_text, add_special_tokens=False)

    # Dataset + Dataloader
    dataset = PretrainDataset(token_ids, block_size=BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ----------------------------------------------------
    # Build model
    # ----------------------------------------------------
    config = MiniGPTConfig(
        vocab_size=len(tokenizer.token_to_id),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=BLOCK_SIZE
    )
    model = MiniGPT(config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ----------------------------------------------------
    # Training loop
    # ----------------------------------------------------
    model.train()
    step = 0
    running_loss = 0.0

    print("Starting training...")

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
                avg = running_loss / 50
                print(f"step {step}/{MAX_STEPS} | loss: {avg:.4f}")
                running_loss = 0.0

            if step >= MAX_STEPS:
                break

    # ----------------------------------------------------
    # Save checkpoint
    # ----------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mini_gpt_pretrained.pt")
    print("Training complete. Saved checkpoint.")


if __name__ == "__main__":
    train()
