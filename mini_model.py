# mini_model.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# Configuration object
# --------------------------------------------------------

@dataclass
class MiniGPTConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1


# --------------------------------------------------------
# Attention
# --------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Classic decoder-only (GPT-style) masked multi-head self-attention.
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.config = config
        self.head_dim = config.d_model // config.n_heads

        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Build a triangular (causal) mask for max_seq_len
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, sequence length, channels

        # Produce Q, K, V
        q = self.q_proj(x)   # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, heads, T, head_dim)
        q = q.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask: no looking ahead
        att = att.masked_fill(
            self.causal_mask[:T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Weighted sum of values
        out = att @ v   # (B, heads, T, head_dim)

        # Reassemble heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


# --------------------------------------------------------
# Feed-forward network
# --------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


# --------------------------------------------------------
# Transformer Block
# --------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# --------------------------------------------------------
# Full MiniGPT model
# --------------------------------------------------------

class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head (weights tied to embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape

        assert T <= self.config.max_seq_len, \
            f"Input length {T} exceeds max_seq_len {self.config.max_seq_len}"

        tok_emb = self.token_embed(input_ids)  # (B, T, d_model)
        pos_emb = self.pos_embed(torch.arange(T, device=input_ids.device))

        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)


        # Training mode
        loss = None
        if targets is not None:
            targets = targets.long()
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            #loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    # ----------------------------------------------------
    # Generation (predict next tokens)
    # ----------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        for _ in range(max_new_tokens):
            # Maybe crop to max_seq_len
            x = input_ids[:, -self.config.max_seq_len:]

            logits, _ = self(x)
            logits = logits[:, -1, :]  # last token

            # temperature scale
            if temperature != 1.0:
                logits = logits / temperature

            # top-k
            if top_k:
                vals, idx = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                logits = mask

            # top-p
            if top_p:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)

                cutoff = cumprobs > top_p
                first_cut = torch.argmax(cutoff.to(torch.int), dim=-1)

                for b in range(logits.size(0)):
                    bad_idx = sorted_idx[b, first_cut[b]:]
                    logits[b, bad_idx] = float("-inf")

            # sample next token
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids
