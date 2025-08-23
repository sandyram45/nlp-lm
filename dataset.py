import torch
from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    def __init__(self, tokens, context_len, variant):
        super(WikiTextDataset, self).__init__()
        self.variant = variant
        self.context_len = context_len
        self.tokens = tokens
        assert self.context_len is not None
        assert self.tokens is not None

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx : idx + self.context_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + self.context_len + 1], dtype=torch.long)
        return x,y

    def __len__(self):
        return len(self.tokens) - self.context_len

