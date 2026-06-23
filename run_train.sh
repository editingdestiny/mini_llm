#!/bin/bash
cd ~/mini-llm
python3 -u - << 'PYEOF' > logs/train_sft_v3.log 2>&1 &
import os, sys, json, random, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from mini_model import MiniGPT, MiniGPTConfig
from mini_tokenizer import BPETokenizer

print("Loading data...", flush=True)
tokenizer = BPETokenizer.load("data/tokenizer.json")

samples = []
with open("data/sft_new/diverse_qa.jsonl") as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples", flush=True)

config = MiniGPTConfig(vocab_size=8192, d_model=1024, n_heads=16, n_layers=4, d_ff=4096, max_seq_len=128, dropout=0.1)
model = MiniGPT(config)

print("Loading pretrained weights...", flush=True)
state = torch.load("checkpoints/mini_gpt_pretrained.pt", map_location="cpu")
model.load_state_dict(state, strict=True)
print("Weights loaded!", flush=True)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

MAX_STEPS = 5000
for step in range(MAX_STEPS):
    item = random.choice(samples)
    text = f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > 127:
        tokens = tokens[:127]
    else:
        tokens = tokens + [0] * (127 - len(tokens))
    
    x = torch.tensor([tokens[:-1]])
    y = torch.tensor([tokens[1:]])
    
    logits, loss = model(x, y)
    if loss is None:
        loss = F.cross_entropy(logits.view(-1, 8192), y.view(-1), ignore_index=0)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 200 == 0:
        print(f"Step {step}/{MAX_STEPS} | Loss: {loss.item():.4f}", flush=True)

torch.save(model.state_dict(), "checkpoints/mini_gpt_sft_v3.pt")
print("Saved to checkpoints/mini_gpt_sft_v3.pt", flush=True)
PYEOF

bash ~/mini-llm/run_train.sh
sleep 10
tail -10 ~/mini-llm/logs/train_sft_v3.log
