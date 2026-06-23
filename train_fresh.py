#!/usr/bin/env python3
"""
Fresh train: pretrain + SFT with correct BPETokenizer.
"""
import os, sys, time, json, random, torch
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'
torch.set_num_threads(2)

sys.path.insert(0, '/app')

from mini_model import MiniGPT, MiniGPTConfig
from mini_tokenizer import BPETokenizer

# ── Config ──────────────────────────────────────────────────────────
TOKENIZER_FILE = "/app/data/tokenizer.json"
PRETRAIN_FILE  = "/app/data/1984_clean.txt"
SFT_FILE       = "/app/data/sft_expanded.jsonl"
CKPT_DIR       = "/app/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 1024
SEQ_LEN  = 256

BATCH_SIZE     = 2
PRETRAIN_STEPS  = 800
PRETRAIN_LR     = 1e-3
LOG_EVERY       = 100
SAVE_EVERY      = 200

SFT_STEPS       = 1200
SFT_LR          = 8e-6  # Very low LR for SFT on small model
SFT_BATCH       = 2
SFT_LOG_EVERY   = 50
SFT_SAVE_EVERY  = 100

SEED = 42
# ───────────────────────────────────────────────────────────────────

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)

set_seed(SEED)

print("="*60)
print("FRESH TRAIN — Memory Efficient")
print("="*60)

# ── Tokenizer ──────────────────────────────────────────────────────
print("\n[1] Loading tokenizer...")
tokenizer = BPETokenizer.load(TOKENIZER_FILE)
vocab_size = len(tokenizer.token_to_id)
eos_id = tokenizer.eos_id
print(f"    Vocab: {vocab_size}, EOS: {eos_id}")

# ── Pretrain data ─────────────────────────────────────────────────
print("\n[2] Loading pretrain data...")
with open(PRETRAIN_FILE) as f:
    text = f.read()

ids = tokenizer.encode(text, add_special_tokens=True)
print(f"    {len(text)} chars -> {len(ids)} tokens")

val_size = int(len(ids) * 0.05)
train_ids = ids[:-val_size]
val_ids   = ids[-val_size:]
print(f"    Train: {len(train_ids)}, Val: {len(val_ids)}")

# ── Model ─────────────────────────────────────────────────────────
print("\n[3] Building model...")
config = MiniGPTConfig(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_seq_len=SEQ_LEN,
    dropout=0.0,
    eos_token_id=eos_id,
)
model = MiniGPT(config)
params = sum(p.numel() for p in model.parameters())
print(f"    Params: {params/1e6:.2f}M, layers={N_LAYERS}, d={D_MODEL}")

optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR)

# ── Pretrain ───────────────────────────────────────────────────────
print(f"\n[4] Pretraining: {PRETRAIN_STEPS} steps, batch={BATCH_SIZE}, seq={SEQ_LEN}")
step = 0
losses = []
t_start = time.time()

while step < PRETRAIN_STEPS:
    idxs = torch.randint(0, len(train_ids) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.tensor([[train_ids[i+j] for j in range(SEQ_LEN)] for i in idxs.tolist()], dtype=torch.long)
    y = torch.tensor([[train_ids[i+j+1] for j in range(SEQ_LEN)] for i in idxs.tolist()], dtype=torch.long)
    
    optimizer.zero_grad()
    logits, loss_val, _, _ = model(x, y)
    loss_val.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    losses.append(loss_val.item())
    step += 1
    
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t_start
        speed = LOG_EVERY / elapsed
        avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
        print(f"    Step {step}/{PRETRAIN_STEPS}: loss={avg:.4f}, speed={speed:.1f} step/s", flush=True)
        t_start = time.time()
    
    if step % SAVE_EVERY == 0:
        ckpt = f"{CKPT_DIR}/pretrain_step{step}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": {k: getattr(config, k) for k in ["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "max_seq_len"]}, "step": step, "loss": avg}, ckpt)
        print(f"    ✓ Saved {ckpt}")

torch.save({"model_state_dict": model.state_dict(), "config": {k: getattr(config, k) for k in ["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "max_seq_len"]}, "step": step, "loss": losses[-1]}, f"{CKPT_DIR}/pretrain_final.pt")
print(f"    ✓ Pretrain done, final loss={losses[-1]:.4f}")

# ── SFT ───────────────────────────────────────────────────────────
print(f"\n[5] SFT: {SFT_STEPS} steps, batch={SFT_BATCH}")
optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR)

with open(SFT_FILE) as f:
    sft_lines = [json.loads(line) for line in f]
print(f"    SFT examples: {len(sft_lines)}")

def encode_sft_example(item):
    """
    Encode SFT example using the sft_dataset.py approach:
    x = [BOS_token, ...text..., EOS_token]
    y = [...text..., EOS_token]  (shifted by 1, no -100 masking)
    
    The model learns to predict every token, including the transition
    from prefix to response. This is critical for learning prompt→response.
    """
    text = f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"
    token_ids = tokenizer.encode(text, add_special_tokens=True)  # adds BOS + EOS
    # x: all tokens except last (model sees everything)
    # y: all tokens except first (predict next token at every position)
    x_ids = token_ids[:-1]
    y_ids = token_ids[1:]
    return x_ids, y_ids

sft_step = 0
sft_losses = []
t_start = time.time()

while sft_step < SFT_STEPS:
    batch = [random.choice(sft_lines) for _ in range(SFT_BATCH)]
    x_list, y_list = [], []
    for item in batch:
        x_ids, y_ids = encode_sft_example(item)
        # Truncate/pad to SEQ_LEN
        x_ids = x_ids[:SEQ_LEN]
        y_ids = y_ids[:SEQ_LEN]
        x_ids += [eos_id] * (SEQ_LEN - len(x_ids))
        y_ids += [-100] * (SEQ_LEN - len(y_ids))
        x_list.append(x_ids)
        y_list.append(y_ids)
    
    x = torch.tensor(x_list, dtype=torch.long)
    y = torch.tensor(y_list, dtype=torch.long)
    
    optimizer.zero_grad()
    logits, loss_val, _, _ = model(x, y)
    loss_val.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    sft_losses.append(loss_val.item())
    sft_step += 1
    
    if sft_step % SFT_LOG_EVERY == 0:
        elapsed = time.time() - t_start
        speed = SFT_LOG_EVERY / elapsed
        avg = sum(sft_losses[-SFT_LOG_EVERY:]) / SFT_LOG_EVERY
        print(f"    SFT {sft_step}/{SFT_STEPS}: loss={avg:.4f}, speed={speed:.1f} step/s", flush=True)
        t_start = time.time()
    
    if sft_step % SFT_SAVE_EVERY == 0:
        ckpt = f"{CKPT_DIR}/sft_step{sft_step}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": {k: getattr(config, k) for k in ["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "max_seq_len"]}, "step": sft_step, "loss": avg}, ckpt)
        print(f"    ✓ Saved {ckpt}")

torch.save({"model_state_dict": model.state_dict(), "config": {k: getattr(config, k) for k in ["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "max_seq_len"]}, "step": SFT_STEPS, "loss": sft_losses[-1]}, f"{CKPT_DIR}/fresh_final.pt")
print(f"    ✓ SFT done, final loss={sft_losses[-1]:.4f}")

# ── Generation test ────────────────────────────────────────────────
print("\n[6] Generation test...")
model.eval()
for prompt_text in ["Who is Winston in 1984?", "What is Big Brother?", "What is Newspeak?"]:
    prompt = f"Instruction: {prompt_text}\n\nResponse:"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        out_ids = model.generate(
            input_tensor, max_new_tokens=60, temperature=0.2,
            top_k=20, top_p=0.9, repetition_penalty=1.1, eos_token_id=eos_id,
        )[0].tolist()
    
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    if "Response:" in text:
        text = text.split("Response:", 1)[-1].strip()
    print(f"  Q: {prompt_text}")
    print(f"  A: {text[:200]!r}")

print("\n" + "="*60)
print("DONE")
print("="*60)