# sft_dataset.py
import json
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """
    Supervised dataset for fine-tuning.

    Each line in the JSONL should have:
        {"instruction": "...", "response": "..."}

    We convert it to:
        <bos>Instruction: ... \n\nResponse: ...<eos>
    """

    def __init__(self, tokenizer, jsonl_path: str, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"

        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Pad / truncate to block size
        if len(token_ids) < self.block_size:
            pad_len = self.block_size - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_id] * pad_len
        else:
            token_ids = token_ids[:self.block_size]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)

        return input_ids, target_ids
