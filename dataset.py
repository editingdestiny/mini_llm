# dataset.py
from __future__ import annotations

import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    Turns pretrain text into a dataset of (input_ids, target_ids)
    for next-token prediction.

    Example:
        For block_size=8 and ids = [1, 2, 3, 4, 5]:

            Inputs:  [1,2,3,4]
            Targets: [2,3,4,5]
    """

    def __init__(self, token_ids, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        # Each sample uses block_size tokens as input
        # so we can extract len - block_size sequences
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]   # inputs
        y = chunk[1:]    # targets
        return x, y
