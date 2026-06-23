# mini_model.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniGPTConfig:
    vocab_size: int
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 4
    d_ff: int = 4096
    max_seq_len: int = 256
    dropout: float = 0.1
    eos_token_id: int | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask)
        self.dropout = nn.Dropout(config.dropout)
        # Per-pass attention capture (cleared at start of each full forward)
        self._attn_logits: Optional[torch.Tensor] = None  # (B, n_heads, T, total_T)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) — input. T is full sequence on first step, T=1 on cached steps.
            k_cache: (B, n_heads, cached_len, head_dim) from previous step (or None).
            v_cache: (B, n_heads, cached_len, head_dim) from previous step (or None).
            pos_offset: starting position for positional embeddings. On cached steps (T=1),
                this should be the number of cached tokens so the new token gets the
                correct position rather than always position 0.

        Returns:
            (output, new_k, new_v) where new_k / new_v are this step's K/V tensors,
            NOT wrapped in a list. These are the per-layer cache values.
        """
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)

        if k_cache is not None and v_cache is not None:
            # Extend: concat cached K/V with new token's K/V
            k_full = torch.cat([k_cache, k], dim=2)   # (B, n_heads, total_len, hdim)
            v_full = torch.cat([v_cache, v], dim=2)
            total_len = k_full.shape[2]
            # Build a (1, 1, T, total_len) causal mask for this step.
            # With KV cache, T=1 (new token only) and total_len = cached + 1.
            # The new token can attend to all cached keys (no future in cache).
            step_mask = torch.ones(1, 1, T, total_len, device=x.device, dtype=torch.bool)
            att = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(~step_mask, float("-inf"))
        else:
            k_full = k
            v_full = v
            total_len = T
            # Standard causal mask (T x T)
            att = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.causal_mask[:total_len, :total_len] == 0, float("-inf"))

        # Capture pre-softmax attention logits for visualisation (only on full forward, T > 1)
        if T > 1:
            self._attn_logits = att.detach().clone()

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v_full  # (B, n_heads, T, hdim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Return the EXTENDED cache (k_full) — includes both cached and new tokens.
        # This ensures each generation step accumulates tokens in the cache.
        # The caller replaces their k_cache[layer] = new_k each step.
        return self.out_proj(out), k_full, v_full


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


class TransformerBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (output, new_k, new_v) for this layer."""
        attn_out, new_k, new_v = self.attn(self.ln1(x), k_cache, v_cache, pos_offset=pos_offset)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_k, new_v


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # NOTE: NOT tying weights — tied weights cause trivial zero-loss at init
        # (the model memorizes by retrieving embedding norms instead of learning)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids,
        targets=None,
        k_cache: Optional[list[torch.Tensor]] = None,
        v_cache: Optional[list[torch.Tensor]] = None,
        pos_offset: int = 0,
    ):
        """
        Forward pass with optional KV caching.

        Without k_cache (training / first generation step):
            input_ids: (B, T) full sequence.
        With k_cache (generation step > 1):
            input_ids: (B, 1) — just the new token.
            k_cache/v_cache: per-layer list of (B, n_heads, seq_len, head_dim).
            pos_offset: starting position for the current tokens. Should be the
                total number of cached tokens so the new token(s) get correct
                position embeddings.
        Returns: (logits, loss, new_k_cache, new_v_cache)
            new_k_cache / new_v_cache are lists indexed by layer.
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len

        tok_emb = self.token_embed(input_ids)
        positions = torch.arange(pos_offset, pos_offset + T, device=input_ids.device)
        pos_emb = self.pos_embed(positions)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        new_k_cache: list[torch.Tensor] = []
        new_v_cache: list[torch.Tensor] = []

        for layer_idx, block in enumerate(self.blocks):
            layer_k_cache = k_cache[layer_idx] if k_cache is not None else None
            layer_v_cache = v_cache[layer_idx] if v_cache is not None else None
            x, block_k, block_v = block(x, layer_k_cache, layer_v_cache, pos_offset=pos_offset)
            new_k_cache.append(block_k)
            new_v_cache.append(block_v)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            targets = targets.long()
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss, new_k_cache, new_v_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=None,
    ):
        """
        Generation with KV caching.

        After the first step, only Q is computed for the new token while K/V
        are extended from the per-layer cache. This avoids recomputing attention
        over the full history each step, giving roughly n_layers speedup.

        repetition_penalty: >1.0 discourages previously generated tokens.
        eos_token_id: early stop when EOS is produced.
        """
        B = input_ids.size(0)

        _eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        finished: list[bool] = [False] * B

        # Per-layer KV caches: k_cache[layer_idx] = (B, n_heads, seq_len, head_dim)
        k_cache: list[torch.Tensor] | None = None
        v_cache: list[torch.Tensor] | None = None

        seen_tokens: list[set[int]] = [set() for _ in range(B)]

        for step in range(max_new_tokens):
            if k_cache is None:
                # First step: full forward over prompt (pos_offset=0)
                logits, _, k_cache, v_cache = self.forward(input_ids, targets=None)
            else:
                # Subsequent steps: forward only last token (T=1).
                # pos_offset = number of cached tokens = total tokens processed so far.
                # input_ids has grown by 1 each step (previous next_id appended),
                # so input_ids.size(1) = prompt_len + step + 1.
                # The cache holds prompt_len + step tokens, so pos_offset = cache_len.
                last_token = input_ids[:, -1:]
                pos_offset = input_ids.size(1) - 1  # = current cache length
                logits, _, k_cache, v_cache = self.forward(
                    last_token, targets=None, k_cache=k_cache, v_cache=v_cache,
                    pos_offset=pos_offset,
                )

            logits = logits[:, -1, :]  # (B, vocab_size)

            # Mask finished items
            for b in range(B):
                if finished[b]:
                    logits[b, :] = float("-inf")
                    logits[b, 0] = 0.0

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    if finished[b]:
                        continue
                    for tok_id in seen_tokens[b]:
                        if logits[b, tok_id] < 0:
                            logits[b, tok_id] *= repetition_penalty
                        else:
                            logits[b, tok_id] /= repetition_penalty

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # top-k
            if top_k and top_k > 0:
                vals, idx = torch.topk(logits, min(top_k, logits.size(-1)))
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                logits = mask

            # top-p (nucleus sampling)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                cutoff_mask = cumprobs > top_p
                first_cut = torch.argmax(cutoff_mask.to(torch.long), dim=-1, keepdim=True)
                first_cut = first_cut.clamp(min=1)
                discard_mask = torch.zeros_like(logits, dtype=torch.bool)
                for b in range(B):
                    discard_mask[b, sorted_idx[b, first_cut[b, 0]:]] = True
                logits = logits.masked_fill(discard_mask, float("-inf"))

            logits = torch.clamp(logits, min=-1e9, max=1e9)

            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
                next_id = next_id.unsqueeze(1) if next_id.dim() == 1 else next_id
            else:
                next_id = torch.multinomial(probs, num_samples=1)

            if _eos is not None:
                for b in range(B):
                    if not finished[b] and next_id[b, 0].item() == _eos:
                        finished[b] = True

            input_ids = torch.cat([input_ids, next_id], dim=1)

            for b in range(B):
                if not finished[b]:
                    seen_tokens[b].add(next_id[b, 0].item())

            if _eos is not None and all(finished):
                break

        return input_ids

    @torch.no_grad()
    def generate_with_stats(
        self,
        tokenizer,
        input_ids,
        max_new_tokens=50,
        temperature=0.2,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=None,
    ):
        """
        Like generate(), but also captures:
          - per_layer_attn: list of (n_heads, T, T) raw attention logits per layer
                           captured during the FIRST (uncached) forward pass over the prompt.
          - kv_cache_shape: list of per-layer KV cache shapes after generation.
          - gen_steps: per-token generation trace showing top-K candidates and chosen token.

        Returns (output_ids, stats_dict).
        """
        B = input_ids.size(0)
        _eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        finished: list[bool] = [False] * B

        k_cache: list[torch.Tensor] | None = None
        v_cache: list[torch.Tensor] | None = None
        seen_tokens: list[set[int]] = [set() for _ in range(B)]

        # ── First forward: capture attention weights + prompt tokens ───
        prompt_len = input_ids.size(1)
        logits_first, _, k_cache, v_cache = self.forward(input_ids, targets=None)

        prompt_tokens: list[str] = []
        for tid in input_ids[0]:
            ts = tokenizer.id_to_token.get(tid.item(), "?")
            ts = ts.replace("</w>", "·").replace("<bos>", "").replace("<eos>", "")
            prompt_tokens.append(ts[:8])

        per_layer_attn: list[torch.Tensor] = []
        for block in self.blocks:
            attn_logits = block.attn._attn_logits
            if attn_logits is not None:
                per_layer_attn.append(attn_logits[0].cpu())
            else:
                per_layer_attn.append(torch.zeros(self.config.n_heads, prompt_len, prompt_len))
            block.attn._attn_logits = None

        # ── Autoregressive generation ─────────────────────────────────
        gen_steps: list[dict] = []

        for step in range(max_new_tokens):
            if step == 0:
                # First step already done — just record its logits
                step_logits = logits_first[0, -1, :].cpu().squeeze(0)
            else:
                last_token = input_ids[:, -1:]
                pos_offset = input_ids.size(1) - 1
                logits_step, _, k_cache, v_cache = self.forward(
                    last_token, targets=None, k_cache=k_cache, v_cache=v_cache,
                    pos_offset=pos_offset,
                )
                step_logits = logits_step[0, -1, :].cpu()

            logits = step_logits.clone()

            # Mask finished
            for b in range(B):
                if finished[b]:
                    logits[b] = float("-inf")
                    logits[b, 0] = 0.0

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    if finished[b]:
                        continue
                    for tok_id in seen_tokens[b]:
                        if logits[tok_id] < 0:
                            logits[tok_id] *= repetition_penalty
                        else:
                            logits[tok_id] /= repetition_penalty

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # top-k
            if top_k and top_k > 0:
                vals, idx = torch.topk(logits, min(top_k, logits.numel()))
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(0, idx, vals)
                logits = mask

            # top-p (nucleus)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs_sorted, dim=-1)
                cutoff_mask = cumprobs > top_p
                first_cut = torch.argmax(cutoff_mask.to(torch.long), keepdim=True).clamp(min=1)
                discard_mask = torch.zeros_like(logits, dtype=torch.bool)
                discard_mask[sorted_idx[first_cut.item():]] = True
                logits = logits.masked_fill(discard_mask, float("-inf"))

            logits = torch.clamp(logits, min=-1e9, max=1e9)
            probs = F.softmax(logits, dim=-1)
            # Ensure 2D for multinomial: (B, V) even when B=1
            probs_2d = probs if probs.dim() == 2 else probs.unsqueeze(0)

            if torch.isnan(probs_2d).any() or torch.isinf(probs_2d).any():
                next_id = torch.argmax(probs_2d, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs_2d, num_samples=1)

            if _eos is not None:
                for b in range(B):
                    if not finished[b] and next_id[b, 0].item() == _eos:
                        finished[b] = True

            input_ids = torch.cat([input_ids, next_id], dim=1)
            for b in range(B):
                if not finished[b]:
                    seen_tokens[b].add(next_id[b, 0].item())

            # Capture top-8 candidates + chosen
            top_vals, top_idx = torch.topk(logits, min(8, logits.numel()))
            step_probs = F.softmax(step_logits, dim=-1)
            candidates = []
            for v, i in zip(top_vals.tolist(), top_idx.tolist()):
                tok_str = tokenizer.id_to_token.get(i, "?")
                tok_str = tok_str.replace("</w>", "·").replace("<bos>", "").replace("<eos>", "")
                candidates.append({"id": i, "token": tok_str[:8], "logit": round(v, 3), "prob": round(step_probs[i].item(), 4)})
            chosen_id = next_id[0, 0].item()
            chosen_str = tokenizer.id_to_token.get(chosen_id, "?")
            chosen_str = chosen_str.replace("</w>", "·").replace("<bos>", "").replace("<eos>", "")
            gen_steps.append({"step": step, "chosen_id": chosen_id, "chosen": chosen_str[:8], "candidates": candidates})

            if _eos is not None and all(finished):
                break

        # ── KV cache summary ──────────────────────────────────────────
        kv_cache_shape: list[dict] = []
        for layer_idx, (k, v) in enumerate(zip(k_cache, v_cache)):
            kv_cache_shape.append({
                "layer": layer_idx,
                "k_shape": list(k.shape),
                "v_shape": list(v.shape),
                "tokens_cached": k.size(2),
            })

        stats = {
            "prompt_len": prompt_len,
            "prompt_tokens": prompt_tokens,
            "per_layer_attn": per_layer_attn,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "kv_cache": kv_cache_shape,
            "gen_steps": gen_steps,
        }

        return input_ids, stats


