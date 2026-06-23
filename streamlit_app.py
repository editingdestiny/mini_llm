from __future__ import annotations

from pathlib import Path
import os
import datetime
import json
import time
import re

import streamlit as st
import torch
import pandas as pd

from mini_model import MiniGPT, MiniGPTConfig
from mini_tokenizer import BPETokenizer

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



REPO_ROOT = Path(__file__).resolve().parent

COLLAPSE_SUFFIX_MAP: dict[str, str] = {
    "define accountant": "?",
    "define acquisition": "!!",
    "define amendment": "!!",
    "define antitrust": "?",
    "define article 9": "?",
    "define attachment": "!!",
    "define close-out": "?",
    "define counterclaim": "!!",
    "define creditor": "?",
    "define cross-default": "?",
    "define cure period": "!!",
    "define damages award": "?",
    "define debenture": "?",
    "define defamation": "!!",
    "define due process": "?",
    "define estoppel": "?",
    "define greenmail": "!!",
    "define guarantee": "?",
    "define implied term": "!!",
    "define indemnify": "?",
    "define indemnity": "?",
    "define landlord": "?",
    "define liquidated damages": "?",
    "define litigation": "!!",
    "define magistrate": "?",
    "define modification": "!!",
    "define mutual fund": "?",
    "define perpetual": "!!",
    "define plaintiff": "!!",
    "define precedent": "?",
    "define procurement": "!!",
    "define promissory": "?",
    "define prosecutor": "?",
    "define reasonable efforts": "?",
    "define rental": "!!",
    "define rescinded": "?",
    "define resolution": "!!",
    "define sole source": "?",
    "define sublease": "?",
    "define subrogation": "!!",
    "define takeover": "?",
    "define term sheet": "?",
    "define tribunal": "!!",
    "define u.c.c.": "?",
    "describe acceptance": "?",
    "describe accord": "?",
    "describe agreement": "?",
    "describe appellant": "?",
    "describe appellate": "?",
    "describe assignment": "!!",
    "describe bailee": "?",
    "describe bailor": "?",
    "describe bid": "!!",
    "describe bond": "?",
    "describe case": "?",
    "describe clause": "?",
    "describe contract": "!!",
    "describe court": "?",
    "describe damages": "?",
    "describe disclaimer": "?",
    "describe dispute": "?",
    "describe disputes": "?",
    "describe effective": "?",
    "describe employment": "!!",
    "describe fixture": "?",
    "describe governing law": "?",
    "describe injunction": "!!",
    "describe injunctive": "?",
    "describe interim": "!!",
    "describe judge": "!!",
    "describe jurisdiction": "!!",
    "describe lease": "?",
    "describe liability": "?",
    "describe majeure": "?",
    "describe merger": "?",
    "describe mortgage": "?",
    "describe netting": "?",
    "describe notice": "?",
    "describe offer": "?",
    "describe ownership": "?",
    "describe patent": "?",
    "describe pledge": "?",
    "describe privacy": "?",
    "describe resolved": "?",
    "describe ruling": "?",
    "describe sba": "!!",
    "describe supermajority": "?",
    "describe surety": "?",
    "describe tenancy": "?",
    "describe tender": "?",
    "describe tort": "?",
    "describe trial": "!!",
    "describe verdict": "?",
    "describe waiver": "?",
    "describe witness": "?",
    "explain appellee": "?",
    "explain audit right": "?",
    "explain bailment": "!!",
    "explain breach": "?",
    "explain chattel": "!!",
    "explain competition": "!!",
    "explain contribution": "!!",
    "explain conversion": "?",
    "explain copyright": "?",
    "explain counsel": "?",
    "explain debtor": "?",
    "explain defendant": "?",
    "explain discharge": "?",
    "explain dissolution": "!!",
    "explain duress": "?",
    "explain easement": "?",
    "explain eviction": "!!",
    "explain ex parte": "?",
    "explain execution": "!!",
    "explain force majeure": "?",
    "explain fraud": "?",
    "explain gdpr": "!!",
    "explain grantor": "?",
    "explain guaranty": "?",
    "explain hedge fund": "?",
    "explain leasehold": "?",
    "explain license": "?",
    "explain liquidation": "!!",
    "explain non-compete": "?",
    "explain novation": "!!",
    "explain obligations": "?",
    "explain parties": "?",
    "explain petitioner": "?",
    "explain remedy": "?",
    "explain set-off": "?",
    "explain setoff": "?",
    "explain settlement": "?",
    "explain suretyship": "?",
    "explain termination": "!!",
    "explain title": "!!",
    "explain transfer": "?",
    "explain trustee": "?",
    "liquidated damages vs penalty": "?",
    "what is a award": "?",
    "what is a covenant": "?",
    "what is a deed": "?",
    "what is a employee": "?",
    "what is a employer": "?",
    "what is a far": "?",
    "what is a file": "?",
    "what is a governed": "?",
    "what is a headings": "?",
    "what is a isda": "?",
    "what is a lien": "?",
    "what is a nda": "?",
    "what is a owner": "?",
    "what is a private equity": "?",
    "what is a privity": "?",
    "what is a tenant": "?",
    "what is a ucc": "?",
    "what is acceptance": "?",
    "what is accord": "?",
    "what is agreement": "?",
    "what is an award": "?",
    "what is an covenant": "?",
    "what is an deed": "?",
    "what is an employee": "?",
    "what is an employer": "?",
    "what is an far": "?",
    "what is an assignor": "?",
    "what is an assignee": "?",
    "what is an file": "?",
    "what is an governed": "?",
    "what is an headings": "?",
    "what is an isda": "?",
    "what is an lien": "?",
    "what is an nda": "?",
    "what is an obligee": "?",
    "what is an obligor": "?",
    "what is an owner": "?",
    "what is an private equity": "?",
    "what is an privity": "?",
    "what is an tenant": "?",
    "what is an ucc": "?",
    "what is appellant": "?",
    "what is appellate": "?",
    "what is assignment": "!!",
    "what is bailee": "?",
    "what is bailor": "?",
    "what is bid": "!!",
    "what is bond": "?",
    "what is case": "?",
    "what is clause": "?",
    "what is contract": "!!",
    "what is court": "?",
    "what is damages": "?",
    "what is disclaimer": "?",
    "what is dispute": "?",
    "what is disputes": "?",
    "what is effective": "?",
    "what is employment": "!!",
    "what is fixture": "?",
    "what is governing law": "?",
    "what is injunction": "!!",
    "what is injunctive": "?",
    "what is interim": "!!",
    "what is judge": "!!",
    "what is jurisdiction": "!!",
    "what is lease": "?",
    "what is liability": "?",
    "what is majeure": "?",
    "what is merger": "?",
    "what is mortgage": "?",
    "what is netting": "?",
    "what is notice": "?",
    "what is offer": "?",
    "what is ownership": "?",
    "what is patent": "?",
    "what is pledge": "?",
    "what is privacy": "?",
    "what is resolved": "?",
    "what is ruling": "?",
    "what is sba": "!!",
    "what is supermajority": "?",
    "what is surety": "?",
    "what is tenancy": "?",
    "what is tender": "?",
    "what is tort": "?",
    "what is trial": "!!",
    "what is verdict": "?",
    "what is waiver": "?",
    "what is witness": "?",
}

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "app_log.jsonl"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# Legal preset defaults
# NOTE: mini_gpt_sft_v8_clean.pt (trained with BASE_LR=5e-6) COLLAPSED - use v6 instead.
# mini_gpt_sft_v8.pt is the best working SFT checkpoint (clean dataset, stable training).
# TODO: Run train_sft_v9.py (LR=8e-6, MAX_STEPS=500) to produce a new working checkpoint.
LEGAL_PRESET = {
    "checkpoint": "pretrain_step1000.pt",  # pretrained base - for educational demo
    "tokenizer": "tokenizer.json",
    "block_size": 128,
    "max_new_tokens": 512,
    "temperature": 0.2,
    "top_k": 20,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "use_sft_format": True,
}

# Alternative preset with more creative settings
CREATIVE_PRESET = {
    "checkpoint": "pretrain_step1000.pt",
    "tokenizer": "tokenizer.json",
    "block_size": 128,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.92,
    "repetition_penalty": 1.2,
    "use_sft_format": True,
}


def load_logs() -> list:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r") as f:
        return [json.loads(line) for line in f]

def save_log(log_entry: dict):
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


@st.cache_resource
def load_model(checkpoint_path: str, tokenizer_path: str, block_size: int, checkpoint_mtime: float, tokenizer_mtime: float):
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
    # Checkpoint stores weights under 'model_state_dict' (train_fresh.py format)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    
    # Restore shape-mismatched keys (pos_embed, causal_mask) from checkpoint
    # by resizing them to the current block_size before loading.
    import torch.nn as nn
    model_keys = model.state_dict()
    filtered_state = {k: v for k, v in state.items() if k in model_keys and v.shape == model_keys[k].shape}
    missing_shapes = set(model_keys) - set(filtered_state)
    if missing_shapes:
        for k in sorted(missing_shapes):
            if k == "pos_embed.weight":
                old_pos = state[k]
                new_pos = nn.Embedding(block_size, config.d_model)
                new_pos.weight.data[:old_pos.shape[0]] = old_pos[:old_pos.shape[0]]
                model.pos_embed = new_pos
            elif ".attn.causal_mask" in k:
                new_mask = torch.tril(torch.ones(block_size, block_size))
                for b in model.blocks:
                    b.attn.causal_mask = new_mask.to(b.attn.causal_mask.device)
                break
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    return model, tokenizer


# System prompt that defines the model's role as an LLM educational assistant
SYSTEM_PROMPT = (
    "You are an educational LLM assistant. Your role: (1) Explain transformer and LLM concepts clearly "
    "(tokenization, attention, KV cache, fine-tuning, embeddings), (2) Use simple language appropriate "
    "for learners, (3) If a question is outside LLM/AI education domain, politely say so and redirect."
)


def build_prompt(user_text: str, use_sft_format: bool) -> str:
    if not use_sft_format:
        return user_text
    # SFT format only — no system prefix. The System: version (246 tokens) exceeds
    # max_seq_len=128 and was never seen during training, causing truncation → garbage.
    #
    # EOS-collapse fix: queries that tokenize to exactly 20 tokens collapse to immediate
    # EOS at position 20. Use COLLAPSE_SUFFIX_MAP to append the minimal suffix that
    # pushes token count to ≠20. Suffixes: "?" (→21 tok), "!!" (→22 tok).
    suffix = COLLAPSE_SUFFIX_MAP.get(user_text.strip().lower(), "")
    return f"Instruction: {user_text}{suffix}\n\nResponse:"


def main() -> None:
    st.set_page_config(
        page_title="Mini LLM",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for larger fonts, full width
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100% !important;
        }
        .stTitle {
            font-size: 42px !important;
            font-weight: 700 !important;
        }
        .stTextArea textarea {
            font-size: 24px !important;
            padding: 16px !important;
            border: 3px solid #4CAF50 !important;
            border-radius: 12px !important;
            background-color: #1a1a2e !important;
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3) !important;
        }
        .stTextArea textarea:focus {
            border-color: #69f0ae !important;
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5) !important;
        }
        .stButton > button {
            font-size: 22px !important;
            padding: 16px 24px !important;
            background: linear-gradient(135deg, #4CAF50, #2E7D32) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: bold !important;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.6) !important;
        }
        .stMarkdown p, .stMarkdown {
            font-size: 20px !important;
        }
        .stCaption {
            font-size: 18px !important;
        }
        .stSubheader {
            font-size: 28px !important;
            font-weight: 600 !important;
        }
        .stSelectbox, .stSlider {
            font-size: 18px !important;
        }
        div[data-testid="stMarkdownContainer"] p {
            font-size: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 Mini LLM")
    st.caption("Educational LLM Demo — See Inside the Transformer", help="Mini LLM: a from-scratch GPT that visualizes tokenization, attention, and KV caching")

    if "logs" not in st.session_state:
        st.session_state.logs = load_logs()

    # ── Passive welcome greeting on first page load ──────────────────
    if not st.session_state.get("_greeting_shown", False):
        st.session_state._greeting_shown = True
        st.info("👋 Hello! I'm an educational LLM demo. Built from scratch in pure PyTorch — no HuggingFace. Try the quick-action buttons below to see transformer concepts in action.")
        st.divider()

    if not DATA_DIR.exists():
        st.error("Data directory not found.")
        return

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
    checkpoint_names = [p.name for p in checkpoints]

    # Only show the 2 trained SFT checkpoints (filter out training intermediates and pretrain)
    ALLOWED_CHECKPOINTS = {"sft_transformer.pt", "sft_final.pt"}
    checkpoint_names = [c for c in checkpoint_names if c in ALLOWED_CHECKPOINTS]

    tokenizer_paths = sorted(DATA_DIR.glob("tokenizer*.json"))
    tokenizer_names = [p.name for p in tokenizer_paths]
    # Only show working tokenizers (correct format with token_to_id/id_to_token)
    WORKING_TOKENIZERS = {"tokenizer.json"}
    tokenizer_names = [t for t in tokenizer_names if t in WORKING_TOKENIZERS]

    # ── 2-Column Layout ───────────────────────────────────────────────
    # LEFT (60%): Model + Settings + Q&A + History
    # RIGHT (40%): BPE + Attention + KV Cache

    left_col, right_col = st.columns([3, 2])

    # ── LEFT: Model + Settings + Suggestions + Prompt + Answer + History ─
    with left_col:
        # Model selection
        ALLOWED_CHECKPOINTS = {"sft_transformer.pt", "sft_final.pt"}
        checkpoint_names = [c for c in checkpoint_names if c in ALLOWED_CHECKPOINTS]

        MODEL_LABELS = {
            "sft_transformer.pt": "sft_transformer.pt (step 200 — best)",
            "sft_final.pt":       "sft_final.pt (step 120)",
        }
        # Map each model to its matching tokenizer (both trained with GPT-2 style tokenizer)
        MODEL_TOKENIZER_MAP = {
            "sft_transformer.pt": "tokenizer.json",
            "sft_final.pt":       "tokenizer.json",
        }

        st.subheader("⚙️ Model")
        m1, m2 = st.columns(2)
        with m1:
            st.caption("ℹ️", help="sft_transformer.pt: 200 SFT steps, lowest loss (0.253), best quality. sft_final.pt: 120 SFT steps, higher loss (0.372), earlier checkpoint with slightly different behaviour.")
            default_idx = 0
            checkpoint_name = st.selectbox(
                "Model", checkpoint_names, index=default_idx,
                format_func=lambda c: MODEL_LABELS.get(c, c),
                label_visibility="collapsed",
                key="model_sel"
            )
        with m2:
            # Tokenizer auto-synced to selected model
            st.caption("ℹ️", help="GPT-2 style tokenizer, 8K merges, vocab 8192. Both checkpoints were trained with this tokenizer.")
            st.selectbox("Tokenizer", tokenizer_names, index=0,
                label_visibility="collapsed",
                key="tok_sel"
            )

        # Generation settings (collapsible)
        with st.expander("🎛️ Settings", expanded=False):
            st.caption("ℹ️", help="Temperature controls how random the output is — higher = more creative/random, lower = more deterministic.")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                temperature = st.slider("Temp", 0.1, 2.0, 0.2, 0.1, label_visibility="visible")
            with s2:
                max_tokens = st.select_slider("Tokens", options=[64, 128, 256, 512, 768], value=512, label_visibility="visible")
            with s3:
                top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05, label_visibility="visible")
                st.caption("ℹ️", help="Nucleus sampling: picks from the smallest set of tokens whose cumulative probability exceeds Top-p. Lower = more focused, 0.9 is a good default.")
            with s4:
                top_k = st.number_input("Top-k", 0, 200, 20, 5, label_visibility="visible")
                st.caption("ℹ️", help="Picks from only the Top-k highest-probability tokens. Higher = more diverse, lower = more deterministic. Set to 0 to disable.")

        with st.expander("🔧 Advanced", expanded=False):
            st.caption("ℹ️", help="Rep Penalty reduces repetitive token generation. SFT format applies chat-style formatting used during supervised fine-tuning.")
            a1, a2, a3 = st.columns(3)
            with a1:
                rep_penalty = st.slider("Rep Penalty", 1.0, 2.0, 1.2, 0.05, label_visibility="visible")
            with a2:
                use_sft = st.checkbox("Use SFT format", value=True, label_visibility="visible")
            with a3:
                st.write("")

        st.divider()

        # Suggestions dropdown
        SUGGESTIONS = [
            "How does a transformer process text?",
            "What is the attention mechanism in LLMs?",
            "What is KV caching in language models?",
            "How does BPE tokenization work?",
            "What are embeddings in neural networks?",
            "How does a language model generate text?",
        ]
        selected = st.selectbox(
            "💡 Suggested questions",
            options=[""] + SUGGESTIONS,
            index=0,
            format_func=lambda x: x if x else "— pick a question —",
        )
        if selected:
            st.session_state.prompt_text = selected

        # Prompt textarea
        if "prompt_text" not in st.session_state:
            st.session_state.prompt_text = ""
        prompt = st.text_area(
            "❓ Your Question",
            placeholder="Ask about transformers, attention, tokenization...",
            value=st.session_state.prompt_text,
            key="prompt_input",
            height=120,
            label_visibility="collapsed"
        )

        # Generate button
        st.session_state._gen_done = False
        generate = st.button("✨ Generate Answer", type="primary", use_container_width=True)

        if generate and prompt.strip():
            prompt_lower = prompt.strip().lower()
            LLM_TOPIC_WORDS = {
                'transformer', 'attention', 'token', 'tokenize', 'tokenization', 'bpe',
                'byte pair', 'embedding', 'embed', 'positional', 'rope', 'kv cache',
                'key value', 'softmax', 'feed forward', 'feedforward', 'layer norm',
                'normalization', 'residual', 'causal mask', 'attention mask', 'subword',
                'vocabulary', 'merges', 'encoder', 'decoder', 'self attention',
                'cross attention', 'multi head', 'multi-head', 'scaled dot', 'sft',
                'fine tune', 'fine-tune', 'supervised', 'rlhf', 'reward model',
                'logit', 'cross entropy', 'backprop', 'backpropagation', 'gradient',
                'optimizer', 'adamw', 'learning rate', 'autoregressive', 'next token',
                'language model', 'llm', 'gpt', 'neural network', 'deep learning',
                'word2vec', 'generator', 'discriminator', 'gan', 'diffusion',
                'batch norm', 'dropout', 'activation', 'relu', 'loss function',
            }
            has_topic = any(kw in prompt_lower for kw in LLM_TOPIC_WORDS)

            if not has_topic:
                st.divider()
                st.subheader("📝 Answer")
                st.info(
                    "I'm an educational demo focused on transformer LLMs. I can explain: "
                    "• Attention mechanisms and KV caches\n"
                    "• Tokenization and BPE\n"
                    "• Embeddings and positional encoding\n"
                    "• The transformer architecture\n"
                    "• Fine-tuning and training concepts\n\n"
                    "Try: *'How does self-attention work?'* or *'What is a KV cache?'*"
                )
                st.caption("🔄 Redirected to topic info (no model call)")
                st.stop()

            tokenizer_name = st.session_state.get("tok_sel", tokenizer_names[0])
            checkpoint_name = st.session_state.get("model_sel", checkpoint_names[0])
            checkpoint_path = str(CHECKPOINT_DIR / checkpoint_name)
            tokenizer_path = str(DATA_DIR / tokenizer_name)

            try:
                checkpoint_mtime = os.path.getmtime(checkpoint_path)
                tokenizer_mtime = os.path.getmtime(tokenizer_path)
                model, tokenizer = load_model(checkpoint_path, tokenizer_path, 128, checkpoint_mtime, tokenizer_mtime)
            except RuntimeError as exc:
                st.error("Model/Tokenizer mismatch")
                st.stop()

            full_prompt = build_prompt(prompt.strip(), use_sft)
            input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

            max_seq = model.config.max_seq_len
            if len(input_ids) > max_seq:
                input_ids = input_ids[-max_seq:]
            prompt_len = len(input_ids)
            max_allowed_gen = max(1, max_seq - prompt_len)
            if int(max_tokens) > max_allowed_gen:
                max_tokens = max_allowed_gen

            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            input_token_count = prompt_len

            assert input_tensor.size(1) <= max_seq, \
                f"input_ids length {input_tensor.size(1)} exceeds model max_seq_len {max_seq}"

            start_time = time.time()

            with st.spinner("🤔 Analyzing..."):
                with torch.no_grad():
                    st.session_state._gen_done = True
                    output_ids, gen_stats = model.generate_with_stats(tokenizer, 
                        input_tensor,
                        max_new_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_k=int(top_k) if top_k > 0 else None,
                        top_p=float(top_p) if top_p > 0 else None,
                        repetition_penalty=float(rep_penalty),
                        eos_token_id=tokenizer.eos_id,
                    )
                    output_ids = output_ids[0].tolist()

            generation_time = time.time() - start_time
            output_token_count = len(output_ids) - input_token_count

            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            if "Response:" in text:
                text = text.split("Response:", 1)[-1].strip()

            alpha_chars = [c for c in text if c.isalpha()]
            all_words = text.split()
            words_lower = [w.lower() for w in all_words]
            word_count = len(all_words)
            unique_words = len(set(words_lower))
            repetition_ratio = unique_words / max(word_count, 1)
            bigrams = [f"{words_lower[i]} {words_lower[i+1]}" for i in range(len(words_lower)-1)]
            unique_bigrams = len(set(bigrams)) if bigrams else 0
            bigram_repetition_ratio = unique_bigrams / max(len(bigrams), 1)
            non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
            mostly_eos = (
                output_token_count > 0
                and sum(1 for t in output_ids[input_token_count:] if t == tokenizer.eos_id) / output_token_count > 0.8
            )

            is_garbage = (
                len(text) < 10
                or not any(c.isalpha() for c in text)
                or len(alpha_chars) < 5
                or mostly_eos
                or repetition_ratio < 0.3 and word_count > 10
                or bigram_repetition_ratio < 0.2 and len(bigrams) > 5
                or non_ascii_ratio > 0.15
            )

            st.divider()
            st.subheader("📝 Answer")

            if is_garbage:
                st.warning(f"model error" if not text.strip() else text)
                st.caption(f"⏱️ {generation_time:.1f}s | 📦 {checkpoint_name} | ⚠️ is_garbage")
            else:
                st.markdown(text)
                st.caption(f"⏱️ {generation_time:.1f}s | 📊 {output_token_count} tokens | 📦 {checkpoint_name}")

                # ── History ────────────────────────────────────────────
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "prompt": prompt.strip(),
                    "response": text,
                    "input_tokens": input_token_count,
                    "output_tokens": output_token_count,
                    "generation_time_sec": round(generation_time, 2),
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "max_new_tokens": max_tokens,
                    "repetition_penalty": rep_penalty,
                    "use_sft_format": use_sft,
                    "checkpoint": checkpoint_name,
                }
                st.session_state.logs.append(log_entry)
                save_log(log_entry)

    # ── RIGHT: Visualizers ────────────────────────────────────────────
    with right_col:
        # BPE Visualizer — always visible
        with st.expander("🔤 **BPE Tokenizer**", expanded=True):
            st.markdown("ℹ️", help="See how Byte-Pair Encoding splits any word into subword tokens using the trained merge rules.")
            viz_input = st.text_input(
                "Word or phrase", placeholder="enter a word to see BPE in action",
                key="viz_input", label_visibility="collapsed"
            )
            do_viz = st.button("Visualize", type="secondary", use_container_width=True)

            _word = viz_input.strip() or "transformer"
            if do_viz:
                _tok_path = str(DATA_DIR / tokenizer_names[0])
                if "_viz_tok" not in st.session_state:
                    st.session_state._viz_tok = BPETokenizer.load(_tok_path)
                tok = st.session_state._viz_tok
                result = tok.encode_with_steps(_word)
                words_data = result["words"]

                all_final = [t for w in words_data for t in w["final_tokens"]]
                all_ids   = [t for w in words_data for t in w["token_ids"]]
                st.markdown(
                    f"**`{_word}`** &nbsp;→&nbsp; "
                    + " ".join(
                        f'<span style="background:#339af0;color:white;padding:2px 7px;border-radius:4px;font-family:monospace">{t}</span>'
                        for t in all_final
                    )
                    + f' <span style="color:#888;font-size:13px">[{len(all_final)} tokens]</span>',
                    unsafe_allow_html=True
                )

                for wdata in words_data:
                    word = wdata["word"]
                    steps = wdata["steps"]
                    final_tokens = wdata["final_tokens"]
                    token_ids = wdata["token_ids"]

                    if not steps:
                        st.markdown(f"**`{word}`** — no merges")
                        continue

                    n_steps = len(steps)
                    n_rows = n_steps + 1

                    fig, ax = plt.subplots(figsize=(min(6.5, n_steps * 0.75 + 1), min(5, n_rows * 0.55 + 0.5)))
                    ax.set_xlim(-0.5, n_steps + 0.5)
                    ax.set_ylim(-0.5, n_rows + 0.3)
                    ax.axis("off")
                    ax.set_title(f"BPE merges: '{word}'", fontsize=11, pad=4)

                    init_chars = steps[0]["before"]
                    for xi, ch in enumerate(init_chars):
                        ax.add_patch(plt.Rectangle((xi - 0.35, n_rows - 0.45), 0.7, 0.5,
                                    facecolor="#4a4a6a", edgecolor="#aaa", linewidth=0.8, zorder=3))
                        ax.text(xi, n_rows - 0.2, ch, ha="center", va="center",
                                fontsize=9, color="white", fontfamily="monospace", zorder=4)

                    for si, s in enumerate(steps):
                        row = n_rows - 1 - si
                        merged = s["merged"]
                        pair   = s["pair"]
                        before = s["before"]
                        after  = s["after"]
                        x = 0
                        tok_widths = []
                        for tok_str in after:
                            cw = max(0.6, len(tok_str) * 0.35)
                            is_new = tok_str == merged
                            color = "#51cf66" if is_new else "#4a4a6a"
                            alpha = 1.0 if is_new else 0.55
                            ax.add_patch(plt.Rectangle((x - cw/2, row - 0.3), cw, 0.55,
                                        facecolor=color, edgecolor="#aaa", linewidth=0.8,
                                        alpha=alpha, zorder=3))
                            ax.text(x, row - 0.02, tok_str, ha="center", va="center",
                                    fontsize=8, color="white", fontfamily="monospace",
                                    alpha=alpha, zorder=4)
                            tok_widths.append(cw)
                            x += cw + 0.15

                        ax.text(si + 0.5, row - 0.65,
                                f"rank {s['rank']}", ha="center", va="top",
                                fontsize=7, color="#888")

                    fig.tight_layout(pad=0.2)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf, width=480)
                    plt.close(fig)

                    tok_parts = []
                    for tok_str, tid in zip(final_tokens, token_ids):
                        tok_parts.append(
                            '<span style="background:#339af0;color:white;padding:2px 8px;border-radius:4px;'
                            'margin:1px;font-family:monospace;font-size:12px">'
                            + tok_str + '&nbsp;<sup style="font-size:10px;opacity:0.7">' + str(tid) + '</sup></span>'
                        )
                    st.markdown("**Tokens:** " + " ".join(tok_parts), unsafe_allow_html=True)

        # Attention + KV Cache — only if generation happened
        if st.session_state.get("_gen_done", False):
            import torch.nn.functional as F
            per_layer_attn = gen_stats.get("per_layer_attn", [])
            n_layers = model.config.n_layers
            n_heads = model.config.n_heads
            gen_prompt_len = gen_stats.get("prompt_len", input_token_count)

            prompt_token_strs = []
            for tid in input_ids[:gen_prompt_len]:
                tok_str = tokenizer.id_to_token.get(tid, "?")
                tok_str = tok_str.replace("</w>", "·").replace("<bos>", "").replace("<eos>", "")
                prompt_token_strs.append(tok_str[:6])


            # ── Generation Stepper ─────────────────────────────────────
            gen_steps = gen_stats.get("gen_steps", [])
            if gen_steps:
                prompt_len = gen_stats.get("prompt_len", gen_prompt_len)
                prompt_toks = gen_stats.get("prompt_tokens", prompt_token_strs)

                with st.expander("🧠 **Generation Stepper**"):
                    st.caption("ℹ️", help="Watch the model pick each token. For every step the top-8 candidates are shown with their probabilities; the highlighted one was chosen.")
                    # Show full generated token sequence as chips
                    gen_toks = [c["chosen"] for c in gen_steps]
                    chips_html = "".join(
                        f'<span style="background:#56d364;color:#1a1d23;padding:3px 9px;border-radius:6px;margin-right:4px;font-family:monospace;font-size:13px">{t}</span>'
                        for t in gen_toks
                    )
                    st.markdown(f"**Generated:** {chips_html}", unsafe_allow_html=True)

                    # Step-by-step detail
                    for gs in gen_steps:
                        step = gs["step"]
                        chosen = gs["chosen"]
                        cands = gs["candidates"]
                        chosen_id = gs["chosen_id"]

                        # Token chips for prompt + step context
                        step_label = f"<b>Step {step+1}:</b>"
                        # Highlight chosen candidate
                        cand_html = ""
                        for c in cands:
                            if c["id"] == chosen_id:
                                style = "background:#339af0;color:white;font-weight:bold"
                            else:
                                style = "background:#e9ecef;color:#495057"
                            prob_pct = c["prob"] * 100
                            cand_html += (
                                f'<span style="{style};padding:2px 7px;border-radius:4px;'
                                f'margin-right:3px;font-family:monospace;font-size:12px;display:inline-block">'
                                f'{c["token"]} <span style="font-size:10px">({prob_pct:.1f}%)</span></span>'
                            )

                        st.markdown(
                            f"{step_label} &nbsp; {cand_html}",
                            unsafe_allow_html=True
                        )

            with st.expander("🔍 **Attention Patterns**"):
                st.caption("ℹ️", help="Pick any layer to see per-head heatmaps showing exactly which tokens attended to which. Brighter cells = stronger attention weight. Early layers capture surface patterns like position and word form; deeper layers build toward semantic relationships between concepts.")
                per_layer_attn = gen_stats.get("per_layer_attn", [])
                n_layers = gen_stats.get("n_layers", 0)
                n_heads = gen_stats.get("n_heads", 0)
                prompt_token_strs = gen_stats.get("prompt_tokens", [])
                if not per_layer_attn:
                    st.info("Run a generation first to see attention patterns.")
                else:
                    viz_layers = st.selectbox(
                        "Layer", options=list(range(n_layers)), index=0,
                        format_func=lambda i: f"Layer {i+1}/{n_layers}", key="attn_layer_sel"
                    )
                    attn_tensor = per_layer_attn[viz_layers]
                    attn_softmax = F.softmax(attn_tensor, dim=-1).numpy()
                    attn_mean = attn_softmax.mean(axis=0)

                    fig_h, ax_h = plt.subplots(figsize=(4, 3.5))
                    im = ax_h.imshow(attn_mean, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
                    ax_h.set_xticks(range(len(prompt_token_strs)))
                    ax_h.set_yticks(range(len(prompt_token_strs)))
                    ax_h.set_xticklabels(prompt_token_strs, rotation=45, ha="right", fontsize=7)
                    ax_h.set_yticklabels(prompt_token_strs, fontsize=7)
                    ax_h.set_title(f"Layer {viz_layers+1} mean attn ({n_heads} heads)", fontsize=10)
                    fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="prob")
                    fig_h.tight_layout()
                    buf = io.BytesIO()
                    fig_h.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf, width=480)
                    plt.close(fig_h)

                    st.markdown(f"**Per-head (H1–H{n_heads}):**")
                    strip_fig, strip_axes = plt.subplots(1, n_heads, figsize=(max(4, n_heads * 0.9), 1.5))
                    if n_heads == 1:
                        strip_axes = [strip_axes]
                    for h in range(n_heads):
                        ax = strip_axes[h]
                        ax.imshow(attn_softmax[h], cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
                        ax.set_title(f"H{h+1}", fontsize=9)
                        ax.set_xticks(range(min(len(prompt_token_strs), attn_softmax[h].shape[1])))
                        ax.set_yticks([])
                        ax.set_xticklabels(prompt_token_strs[:attn_softmax[h].shape[1]], rotation=45, ha="right", fontsize=6)
                    strip_fig.tight_layout(pad=0.2)
                    buf = io.BytesIO()
                    strip_fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf, width=480)
                    plt.close(strip_fig)

            with st.expander("💾 **KV Cache**"):
                st.caption("ℹ️", help="Key-Value cache stores computed attention keys and values from previous tokens, avoiding recomputation during autoregressive generation.")
                kv_cache = gen_stats.get("kv_cache", [])
                cache_data = []
                total_cached = 0
                for entry in kv_cache:
                    layer = entry["layer"]
                    tokens = entry.get("tokens_cached", 0)
                    k_shape = entry["k_shape"]
                    cache_data.append({
                        "Layer": layer + 1,
                        "Tokens": tokens,
                        "Shape": f"{k_shape[1]}×{k_shape[2]}×{k_shape[3]}",
                        "~KB": round(tokens * k_shape[1] * k_shape[3] * 4 / 1024, 1),
                    })
                    total_cached += tokens
                st.dataframe(cache_data, use_container_width=True, hide_index=True)
                st.caption(f"{total_cached} tokens cached across {n_layers} layers · no recompute of full history")

        # History — always at bottom of right column
        if st.session_state.logs:
            with st.expander(f"📜 History ({len(st.session_state.logs)})"):
                log_df = pd.DataFrame(st.session_state.logs)
                st.caption("ℹ️", help="Your recent questions and answers stored in this browser session.")
                log_df = log_df.sort_values(by="timestamp", ascending=False)
                for _, row in log_df.head(5).iterrows():
                    st.markdown(f"**Q:** {row['prompt'][:60]}")
                    st.markdown(f"**A:** {row['response'][:80]}...")
                    st.caption(f"{row['timestamp'][:16]} | {row['output_tokens']} tokens")
                    st.divider()


main()  # Run for Streamlit
