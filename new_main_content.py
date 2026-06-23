    # ── 2-Column Layout ───────────────────────────────────────────────
    # LEFT (60%): Model + Settings + Q&A + History
    # RIGHT (40%): BPE + Attention + KV Cache

    left_col, right_col = st.columns([3, 2])

    # ── LEFT: Model + Settings + Suggestions + Prompt + Answer + History ─
    with left_col:
        # Model selection
        ALLOWED_CHECKPOINTS = {"sft_transformer.pt", "sft_final.pt"}
        checkpoint_names = [c for c in checkpoint_names if c in ALLOWED_CHECKPOINTS]
        st.subheader("⚙️ Model")
        m1, m2 = st.columns(2)
        with m1:
            default_idx = 0
            checkpoint_name = st.selectbox("Model", checkpoint_names, index=default_idx, label_visibility="collapsed")
        with m2:
            st.selectbox("Tokenizer", tokenizer_names, index=0, label_visibility="collapsed")

        # Generation settings
        st.subheader("🎛️ Settings")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            temperature = st.slider("Temp", 0.1, 2.0, 0.2, 0.1, label_visibility="visible")
        with s2:
            max_tokens = st.select_slider("Tokens", options=[64, 128, 256, 512, 768], value=256, label_visibility="visible")
        with s3:
            top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05, label_visibility="visible")
        with s4:
            top_k = st.number_input("Top-k", 0, 200, 20, 5, label_visibility="visible")

        st.subheader("🔧 Advanced")
        a1, a2, a3 = st.columns(3)
        with a1:
            rep_penalty = st.slider("Rep Penalty", 1.0, 2.0, 1.1, 0.05, label_visibility="visible")
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

            tokenizer_name = tokenizer_names[0]
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
                    output_ids, gen_stats = model.generate_with_stats(
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
            viz_input = st.text_input(
                "Word or phrase", placeholder="e.g. transformer",
                key="viz_input", label_visibility="collapsed"
            )
            do_viz = st.button("Visualize", type="secondary", use_container_width=True)

            if do_viz and viz_input.strip():
                _tok_path = str(DATA_DIR / tokenizer_names[0])
                if "_viz_tok" not in st.session_state:
                    st.session_state._viz_tok = BPETokenizer.load(_tok_path)
                tok = st.session_state._viz_tok
                result = tok.encode_with_steps(viz_input)
                words_data = result["words"]

                all_final = [t for w in words_data for t in w["final_tokens"]]
                all_ids   = [t for w in words_data for t in w["token_ids"]]
                st.markdown(
                    f"**`{viz_input}`** &nbsp;→&nbsp; "
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

                    fig, ax = plt.subplots(figsize=(min(5.5, n_steps * 0.65 + 1), min(4, n_rows * 0.5 + 0.4)))
                    ax.set_xlim(-0.5, n_steps + 0.5)
                    ax.set_ylim(-0.5, n_rows + 0.3)
                    ax.axis("off")
                    ax.set_title(f"BPE merges: '{word}'", fontsize=8, pad=4)

                    init_chars = steps[0]["before"]
                    for xi, ch in enumerate(init_chars):
                        ax.add_patch(plt.Rectangle((xi - 0.35, n_rows - 0.45), 0.7, 0.5,
                                    facecolor="#4a4a6a", edgecolor="#aaa", linewidth=0.8, zorder=3))
                        ax.text(xi, n_rows - 0.2, ch, ha="center", va="center",
                                fontsize=7, color="white", fontfamily="monospace", zorder=4)

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
                                    fontsize=6, color="white", fontfamily="monospace",
                                    alpha=alpha, zorder=4)
                            tok_widths.append(cw)
                            x += cw + 0.15

                        ax.text(si + 0.5, row - 0.65,
                                f"rank {s['rank']}", ha="center", va="top",
                                fontsize=5, color="#888")

                    fig.tight_layout(pad=0.2)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
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

            with st.expander("🔍 **Attention Patterns**"):
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
                ax_h.set_xticklabels(prompt_token_strs, rotation=45, ha="right", fontsize=5)
                ax_h.set_yticklabels(prompt_token_strs, fontsize=5)
                ax_h.set_title(f"Layer {viz_layers+1} mean attn ({n_heads} heads)", fontsize=7)
                fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="prob")
                fig_h.tight_layout()
                buf = io.BytesIO()
                fig_h.savefig(buf, format='png', dpi=80, bbox_inches='tight')
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
                    ax.set_title(f"H{h+1}", fontsize=6)
                    ax.set_xticks(range(min(len(prompt_token_strs), attn_softmax[h].shape[1])))
                    ax.set_yticks([])
                    ax.set_xticklabels(prompt_token_strs[:attn_softmax[h].shape[1]], rotation=45, ha="right", fontsize=4)
                strip_fig.tight_layout(pad=0.2)
                buf = io.BytesIO()
                strip_fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                buf.seek(0)
                st.image(buf, width=480)
                plt.close(strip_fig)

            with st.expander("💾 **KV Cache**"):
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
                log_df = log_df.sort_values(by="timestamp", ascending=False)
                for _, row in log_df.head(5).iterrows():
                    st.markdown(f"**Q:** {row['prompt'][:60]}")
                    st.markdown(f"**A:** {row['response'][:80]}...")
                    st.caption(f"{row['timestamp'][:16]} | {row['output_tokens']} tokens")
                    st.divider()
