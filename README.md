# Mini-LLM

Custom GPT-style language model served via Streamlit.

**Live:** https://mini-llm.sd-ai.co.uk

## Architecture

- Model: custom transformer with GPT2-style BPE tokenizer (~8192 tokens)
- Checkpoints: `checkpoints/` (not in git — large binary files)
- Training data: `data/phase4/` (not in git)
- Serving: Streamlit on port 8502, reverse-proxied via Traefik

## Development

```bash
# Local dev
cd /home/sd22750/mini-llm
source venv/bin/activate
streamlit run streamlit_app.py

# Build Docker image locally
docker build -t mini-llm:local .

# Run with docker-compose
docker-compose up -d
```

## SDLC

- Push to `main` → GitHub Actions builds + pushes image to GHCR
- CI deploys to server via SSH + smoke tests the endpoint
- Image tag: `ghcr.io/editingdestiny/mini-llm:<sha>`

## Training

```bash
# Continue training from checkpoint
python continue_train.py --checkpoint checkpoints/sft_final.pt

# Fresh training
python train_fresh.py
```
