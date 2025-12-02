# 🤖 Mini LLM

A lightweight, educational implementation of a Large Language Model from scratch using PyTorch. This project demonstrates the complete pipeline of building, training, and fine-tuning a transformer-based language model.

## 🌟 Features

- **Custom Tokenizer**: BPE (Byte Pair Encoding) tokenizer trained from scratch
- **Transformer Architecture**: GPT-style decoder-only model with multi-head attention
- **Two-Stage Training**:
  - **Pretraining**: Next-token prediction on raw text corpus
  - **Supervised Fine-Tuning (SFT)**: Instruction-following using question-answer pairs
- **Text Generation**: Sample text from your trained model with customizable parameters

## 📁 Project Structure

```
mini-llm/
├── mini_tokenizer.py      # Custom BPE tokenizer implementation
├── mini_model.py          # Transformer model architecture
├── dataset.py             # Dataset loader for pretraining
├── sft_dataset.py         # Dataset loader for supervised fine-tuning
├── train_tokenizer.py     # Script to train the tokenizer
├── train_pretrain.py      # Pretraining script
├── train_sft.py           # Fine-tuning script
├── generate_text.py       # Text generation from trained model
├── test_model.py          # Model testing utilities
├── data/
│   ├── raw/
│   │   └── pretrain.txt   # Raw text corpus for pretraining
│   ├── sft/
│   │   └── instructions.jsonl  # Instruction-response pairs
│   └── tokenizer.json     # Trained tokenizer vocabulary
└── checkpoints/
    ├── mini_gpt_pretrained.pt  # Pretrained model weights
    └── mini_gpt_sft.pt         # Fine-tuned model weights
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch
```

### Training Pipeline

#### 1. Train the Tokenizer

First, train a BPE tokenizer on your text corpus:

```bash
python train_tokenizer.py
```

This creates a vocabulary and saves it to `data/tokenizer.json`.

#### 2. Pretrain the Model

Train the model on next-token prediction:

```bash
python train_pretrain.py
```

The model learns general language patterns and saves checkpoints to `checkpoints/mini_gpt_pretrained.pt`.

#### 3. Supervised Fine-Tuning

Fine-tune the pretrained model on instruction-following tasks:

```bash
python train_sft.py
```

This aligns the model to follow instructions using the data in `data/sft/instructions.jsonl`.

#### 4. Generate Text

Generate text from your trained model:

```bash
python generate_text.py
```

## 🏗️ Model Architecture

- **Type**: GPT-style Decoder-only Transformer
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Layers**: 4
- **Vocabulary Size**: 512 tokens
- **Context Length**: 128 tokens

## 📊 Training Configuration

### Pretraining
- **Objective**: Next-token prediction
- **Batch Size**: 16
- **Epochs**: 20
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW

### Fine-Tuning
- **Objective**: Instruction following
- **Batch Size**: 4
- **Epochs**: 10
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW

## 🎯 Use Cases

This project is perfect for:
- **Learning**: Understanding transformer architecture and LLM training pipeline
- **Experimentation**: Testing new training techniques on a small scale
- **Education**: Teaching others about language model fundamentals
- **Prototyping**: Quick iterations before scaling to larger models

## 🔧 Customization

### Modify Model Size

Edit `mini_model.py` to change:
- `d_model`: Embedding dimension
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer blocks
- `max_len`: Maximum sequence length

### Add Your Own Data

- **Pretraining**: Add text files to `data/raw/pretrain.txt`
- **Fine-tuning**: Add instruction pairs to `data/sft/instructions.jsonl` in the format:
  ```json
  {"instruction": "Question here", "response": "Answer here"}
  ```

## 📈 Monitoring Training

Both training scripts output:
- Loss per epoch
- Training progress
- Model checkpoints

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Built with PyTorch and inspired by the GPT architecture from "Attention Is All You Need" and subsequent transformer models.

---

**Note**: This is a educational implementation designed for learning purposes. For production use, consider established frameworks like Hugging Face Transformers.
