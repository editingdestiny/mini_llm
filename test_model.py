# test_model.py
import torch
from mini_model import MiniGPT, MiniGPTConfig

config = MiniGPTConfig(
    vocab_size=4096,
    d_model=128,
    n_heads=4,
    n_layers=2,
    max_seq_len=512,
)

model = MiniGPT(config)
torch.manual_seed(42)
x = torch.randint(0, config.vocab_size, (8, 128))
#x = torch.randint(0, config.vocab_size, (2, 20))
logits, loss = model(x, x)

print("logits:", logits.shape)
print("loss:", loss.item())
print("targets min/max:", x.min().item(), x.max().item())
print("targets dtype:", x.dtype)
print("loss is", loss)
print("loss type:", type(loss))