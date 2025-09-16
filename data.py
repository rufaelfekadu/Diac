import torch
from torch.utils.data import Dataset, DataLoader
from utils import map_data, map_asr_data, CHARACTERS_MAPPING, CLASSES_MAPPING
from typing import List, Tuple, Optional

class TextDataset(Dataset):
    """
    Dataset class for text-only data.
    """
    def __init__(self, lines: List[str]):
        self.lines = lines

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        line = self.lines[idx]
        X, Y = map_data([line])
        return X[0], Y[0]

class TextAudioDataset(Dataset):
    """
    Dataset class for text + audio (ASR) data.

    """
    def __init__(self, lines: List[str], asr_lines: List[str], expanded_vocabulary: dict):
        self.lines = lines
        self.asr_lines = asr_lines
        self.expanded_vocabulary = expanded_vocabulary

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        line = self.lines[idx]
        asr_line = self.asr_lines[idx]
        X, Y = map_data([line])
        X_asr = map_asr_data([asr_line], self.expanded_vocabulary)
        return X[0], X_asr[0], Y[0]

def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader with custom collate function.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function for variable length sequences.
    """
    if len(batch[0]) == 2:  # Text-only
        X_batch = [torch.tensor(item[0], dtype=torch.long) for item in batch]
        Y_batch = [torch.tensor(item[1], dtype=torch.long) for item in batch]
        X_batch = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True, padding_value=CHARACTERS_MAPPING.get('<PAD>', 0))
        Y_batch = torch.nn.utils.rnn.pad_sequence(Y_batch, batch_first=True, padding_value=CLASSES_MAPPING.get('<PAD>', 0))
        return X_batch, Y_batch
    elif len(batch[0]) == 3:  # Text + Audio
        X_batch = [torch.tensor(item[0], dtype=torch.long) for item in batch]
        X_asr_batch = [torch.tensor(item[1], dtype=torch.long) for item in batch]
        Y_batch = [torch.tensor(item[2], dtype=torch.long) for item in batch]
        X_batch = torch.nn.utils.rnn.pad_sequence(X_batch, batch_first=True, padding_value=CHARACTERS_MAPPING.get('<PAD>', 0))
        X_asr_batch = torch.nn.utils.rnn.pad_sequence(X_asr_batch, batch_first=True, padding_value=CHARACTERS_MAPPING.get('<PAD>', 0))
        Y_batch = torch.nn.utils.rnn.pad_sequence(Y_batch, batch_first=True, padding_value=CLASSES_MAPPING.get('<PAD>', 0))
        return X_batch, X_asr_batch, Y_batch
    else:
        raise ValueError("Unexpected batch structure")
